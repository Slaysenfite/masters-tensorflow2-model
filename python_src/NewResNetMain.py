import gc
import sys
import time
from datetime import timedelta

import tensorflow as tf
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2

from configurations.DataSet import cbis_ddsm_data_set as data_set
from configurations.TrainingConfig import IMAGE_DIMS, create_required_directories, hyperparameters, create_callbacks
from networks.NetworkHelper import create_classification_layers, compile_with_regularization, generate_heatmap
from training_loops.CustomCallbacks import RunMetaHeuristicOnPlateau
from training_loops.CustomTrainingLoop import training_loop
from utils.ScriptHelper import read_cmd_line_args, evaluate_classification_model, evaluate_meta_model

print('Python version: {}'.format(sys.version))
print('Tensorflow version: {}\n'.format(tf.__version__))
print('[BEGIN] Start script...\n')
hyperparameters, opt, data_set = read_cmd_line_args(hyperparameters, data_set)
print(' Dataset: {}\n'.format(data_set.name))
print(' Image dimensions: {}\n'.format(IMAGE_DIMS))
print(hyperparameters.report_hyperparameters())

print('[INFO] Creating required directories...')
create_required_directories()
gc.enable()

print('FIRST PASS')

print('[INFO] Loading images...')
train_x, test_x, train_y, test_y = data_set.split_data_set(IMAGE_DIMS,
                                                           subset=None,
                                                           segment=hyperparameters.dataset_segment)
loss, train_y, test_y = data_set.get_dataset_labels(train_y, test_y)

if hyperparameters.preloaded_weights:
    print('[INFO] Loading imagenet weights')
    weights = 'imagenet'
else:
    weights = None

model = ResNet50V2(
    include_top=False,
    weights=weights,
    input_shape=IMAGE_DIMS,
    classes=len(data_set.class_names))
model = create_classification_layers(base_model=model,
                                     classes=len(data_set.class_names),
                                     dropout_prob=hyperparameters.dropout_prob)

# Compile model
compile_with_regularization(model=model,
                            loss='binary_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy'],
                            regularization_type='l2',
                            l2=hyperparameters.l2)

# Setup callbacks
callbacks = create_callbacks(hyperparameters)

# train the network
start_time = time.time()

if hyperparameters.meta_heuristic != 'none':
    meta_callback = RunMetaHeuristicOnPlateau(
        X=train_x, y=train_y, meta_heuristic=hyperparameters.meta_heuristic, population_size=30, iterations=10,
        monitor='val_loss', patience=6, verbose=1, mode='min', min_delta=0.001, cooldown=4)
    callbacks.append(meta_callback)

H = model.fit(train_x, train_y, batch_size=hyperparameters.batch_size, validation_data=(test_x, test_y),
              steps_per_epoch=len(train_x) // hyperparameters.batch_size, epochs=hyperparameters.epochs,
              callbacks=callbacks)

time_taken = timedelta(seconds=(time.time() - start_time))

generate_heatmap(model, test_x, 10, 0, hyperparameters, '_1st_pass')
generate_heatmap(model, test_x, 10, 1, hyperparameters, '_1st_pass')

# evaluate the network
evaluate_classification_model(model, 'ResNet50Class', hyperparameters, data_set, H, time_taken, test_x, test_y)

print('META-HEURISTIC')

H = training_loop(model, hyperparameters, train_x, train_y, test_x, test_y,
                  meta_heuristic=hyperparameters.meta_heuristic)

print('EVALUATION')

generate_heatmap(model, test_x, 10, 0, hyperparameters, '_meta_pass')
generate_heatmap(model, test_x, 10, 1, hyperparameters, '_meta_pass')

# evaluate the network
evaluate_meta_model(model, 'ResNet50Meta', hyperparameters, data_set, test_x, test_y)

print('[END] Finishing script...\n')
