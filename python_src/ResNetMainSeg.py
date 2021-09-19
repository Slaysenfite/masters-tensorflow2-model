import gc
import sys
import time
from datetime import timedelta

import tensorflow as tf
from IPython.core.display import clear_output
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.losses import CategoricalHinge
from tensorflow.python.keras.metrics import MeanIoU
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from configurations.DataSet import cbis_seg_data_set as data_set
from configurations.TrainingConfig import IMAGE_DIMS, hyperparameters, output_dir, MODEL_OUTPUT, \
    create_required_directories, create_callbacks
from metrics.MetricsReporter import MetricReporter
from metrics.MetricsUtil import iou_coef, dice_coef
from networks.NetworkHelper import compile_with_regularization, generate_heatmap, create_classification_layers, \
    create_segmentation_layers
from training_loops.CustomCallbacks import RunMetaHeuristicOnPlateau
from training_loops.CustomTrainingLoop import training_loop
from training_loops.OptimizerHelper import calc_seg_fitness
from utils.ImageLoader import load_seg_images, supplement_seg_training_data
from utils.ScriptHelper import create_file_title, read_cmd_line_args, generate_script_report
from utils.SegScriptHelper import show_predictions

print('Python version: {}'.format(sys.version))
print('Tensorflow version: {}\n'.format(tf.__version__))
print('[BEGIN] Start script...\n')
hyperparameters, opt, data_set = read_cmd_line_args(hyperparameters, data_set)
print(' Image dimensions: {}\n'.format(IMAGE_DIMS))
print(hyperparameters.report_hyperparameters())

print('[INFO] Creating required directories...')
create_required_directories()
gc.enable()

print('[INFO] Loading images...')
train_y, roi_train_labels = load_seg_images(data_set, path_suffix='roi',
                                            image_dimensions=(IMAGE_DIMS[0], IMAGE_DIMS[1], 1), subset='Training')
test_y, roi_labels = load_seg_images(data_set, path_suffix='roi', image_dimensions=(IMAGE_DIMS[0], IMAGE_DIMS[1], 1),
                                     subset='Test')

train_x, train_labels = load_seg_images(data_set, image_dimensions=IMAGE_DIMS, subset='Training')
test_x, test_labels = load_seg_images(data_set, image_dimensions=IMAGE_DIMS, subset='Test')

if hyperparameters.augmentation:
    print('[INFO] Augmenting data set')
    aug = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=10,
        zoom_range=0.05,
        fill_mode='nearest')

    train_x, train_y = supplement_seg_training_data(aug, train_x, train_y, roi_labels)

print('[INFO] Training data shape: ' + str(train_x.shape))
print('[INFO] Training label shape: ' + str(train_y.shape))

clear_output()

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

if hyperparameters.weights_of_experiment_id is not None:
    path_to_weights = '{}{}.h5'.format(MODEL_OUTPUT, hyperparameters.weights_of_experiment_id)
    print('[INFO] Loading weights from {}'.format(path_to_weights))
    model.load_weights(path_to_weights)

model = create_segmentation_layers(model)
# Compile model
compile_with_regularization(model=model,
                            loss=CategoricalHinge(),
                            optimizer=opt,
                            metrics=['accuracy', MeanIoU(num_classes=len(data_set.class_names))],
                            regularization_type='l2',
                            l2=hyperparameters.l2)

# Setup callbacks
callbacks = create_callbacks(hyperparameters)

if hyperparameters.meta_heuristic != 'none':
    meta_callback = RunMetaHeuristicOnPlateau(
        X=train_x, y=train_y, meta_heuristic=hyperparameters.meta_heuristic, population_size=30, iterations=10,
        fitness_function=calc_seg_fitness, monitor='val_loss', patience=4, verbose=1, mode='max',
        min_delta=0.05, cooldown=3)
    callbacks.append(meta_callback)

start_time = time.time()

if hyperparameters.tf_fit:

    H = model.fit(train_x, train_y, batch_size=hyperparameters.batch_size, validation_data=(test_x, test_y),
                  steps_per_epoch=len(train_x) // hyperparameters.batch_size, epochs=hyperparameters.epochs,
                  callbacks=callbacks)
else:
    H = training_loop(model, opt, hyperparameters, train_x, train_y, test_x, test_y,
                      meta_heuristic=hyperparameters.meta_heuristic,
                      fitness_function=calc_seg_fitness, task='segmentation')

show_predictions(model, test_x, test_y, 2, output_dir + 'segmentation/' + hyperparameters.experiment_id + '_pred2.png')
show_predictions(model, test_x, test_y, 12,
                 output_dir + 'segmentation/' + hyperparameters.experiment_id + '_pred12.png')
show_predictions(model, test_x, test_y, 22,
                 output_dir + 'segmentation/' + hyperparameters.experiment_id + '_pred22.png')

# evaluate the network
print('[INFO] evaluating network...')

predictions = model.predict(test_x, batch_size=32)

print('[INFO] generating metrics for first pass...')

file_title = create_file_title('UNetSeg2', hyperparameters)

acc = model.evaluate(test_x, test_y)
output = str(model.metrics_names) + '\n'
output += str(acc) + '\n'
output += 'IOU: {}\n'.format(iou_coef(test_y, predictions))
output += 'Dice: {}\n'.format(dice_coef(test_y, predictions))

with open(output_dir + file_title + '_metrics.txt', 'w+') as text_file:
    text_file.write(output)

""" SECOND PASS"""

loss, train_labels, test_labels = data_set.get_dataset_labels(train_labels, test_labels)

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

H = model.fit(train_x, train_labels, batch_size=hyperparameters.batch_size, validation_data=(test_x, test_labels),
              steps_per_epoch=len(train_labels) // hyperparameters.batch_size, epochs=hyperparameters.epochs,
              callbacks=callbacks)

time_taken = timedelta(seconds=(time.time() - start_time))

print('[INFO] evaluating network...')

acc = model.evaluate(test_x, test_labels)
print(str(model.metrics_names))
print(str(acc))

predictions = model.predict(test_x)

print('[INFO] generating metrics for second pass...')

model.save(filepath=MODEL_OUTPUT + file_title + '.h5', save_format='h5')

generate_script_report(H, model, test_x, test_labels, predictions, time_taken, data_set, hyperparameters, file_title)

reporter = MetricReporter(data_set.name, file_title)
cm1 = confusion_matrix(test_labels.argmax(axis=1), predictions.argmax(axis=1))
reporter.plot_confusion_matrix(cm1, classes=data_set.class_names,
                               title='Confusion matrix, without normalization')

reporter.plot_roc(data_set.class_names, test_labels, predictions)

reporter.plot_network_metrics(H, file_title)

generate_heatmap(model, test_x, 10, 0, hyperparameters)
generate_heatmap(model, test_x, 10, 1, hyperparameters)

print('[END] Finishing script...\n')
