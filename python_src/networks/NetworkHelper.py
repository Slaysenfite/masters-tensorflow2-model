import re

from tensorflow.python.keras import regularizers
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Flatten, Conv2D, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.layers.pooling import AveragePooling2D
from tf_explain.core import GradCAM

from configurations.TrainingConfig import SEGMENTATION_OUTPUT, HEATMAP_OUTPUT


def create_classification_layers(base_model, classes, dropout_prob=0.3, kernel_initializer='he_uniform',
                                 layers_removed=-1):
    x = GlobalAveragePooling2D(name='avg_pool')(base_model.layers[layers_removed].output)
    x = Dense(classes, activation='softmax',
                     name='predictions')(x)
    return Model(inputs=base_model.inputs, outputs=x)


def create_segmentation_layers(base_model, layers_removed=-1):
    x = Conv2D(1, (1, 1), padding="same")(base_model.layers[layers_removed].output)
    x = Activation("sigmoid")(x)
    return Model(inputs=base_model.inputs, outputs=x)


def compile_with_regularization(model,
                                loss,
                                optimizer,
                                metrics,
                                add_l_type_reg=True,
                                attrs=['kernel_regularizer'],
                                regularization_type='l2',
                                l1=1.e-4,
                                l2=1.e-4):
    regularizer = get_regularizer(regularization_type, l1, l2)
    for layer in model.layers:
        if isinstance(layer, Conv2D) and add_l_type_reg is True:
            for attr in attrs:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)
                    # print('[INFO] Adding {} {} to {}'.format(regularization_type, attr, layer.name))
    # compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


def get_regularizer(regularization_type, l1, l2):
    if regularization_type == 'l1':
        regularizer = regularizers.l1(l1)
    elif regularization_type == 'l2':
        regularizer = regularizers.l2(l2)
    elif regularization_type == 'l1_l2':
        regularizer = regularizers.l1_l2(l1, l2)
    else:
        print('[ERROR] Regularizer not recognised. Defaulting to L2.')
        regularizer = regularizers.l2()
    return regularizer


def generate_heatmap(model, images, size, class_index, hyperparameters, path_suffix=''):
    # Start explainer
    explainer = GradCAM()
    for i in range(size):
        data = ([images[i]], None)
        grid = explainer.explain(data, model, class_index=class_index)
        explainer.save(grid, HEATMAP_OUTPUT,
                       "{}_{}_class_{}{}.png".format(hyperparameters.experiment_id, i, class_index, path_suffix))


def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                    {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            if insert_layer_name:
                new_layer.name = insert_layer_name
                setattr(new_layer, '_name', insert_layer_name)
            else:
                setattr(new_layer, '_name', '{}_{}'.format(layer.name,
                                                           new_layer.name))
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                                layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)


def dropout_layer_factory():
    return Dropout(rate=0.25, name='dropout')
