import gc
import sys
import time
from datetime import timedelta

import tensorflow as tf
from IPython.core.display import clear_output
from tensorflow.python.keras.losses import BinaryCrossentropy, CategoricalHinge
from tensorflow.python.keras.metrics import MeanIoU
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from configurations.DataSet import cbis_seg_data_set as data_set
from configurations.TrainingConfig import IMAGE_DIMS, hyperparameters, output_dir, MODEL_OUTPUT, \
    create_required_directories, create_callbacks
from metrics.MetricsUtil import iou_coef, dice_coef
from networks.NetworkHelper import compile_with_regularization, generate_heatmap
from networks.UNet import build_pretrained_unet
from networks.UNetSeg import unet_seg
from training_loops.CustomCallbacks import RunMetaHeuristicOnPlateau
from training_loops.CustomTrainingLoop import training_loop
from training_loops.OptimizerHelper import calc_seg_fitness
from utils.ImageLoader import load_seg_images, supplement_seg_training_data
from utils.ScriptHelper import create_file_title, read_cmd_line_args
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
    model = build_pretrained_unet(IMAGE_DIMS, data_set.get_num_classes())
else:
    model = unet_seg(IMAGE_DIMS)

if hyperparameters.weights_of_experiment_id is not None:
    path_to_weights = '{}{}.h5'.format(MODEL_OUTPUT, hyperparameters.weights_of_experiment_id)
    print('[INFO] Loading weights from {}'.format(path_to_weights))
    model.load_weights(path_to_weights)

# Compile model
compile_with_regularization(model=model,
                            loss=BinaryCrossentropy(),
                            optimizer=opt,
                            metrics=['accuracy'],
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
time_taken = timedelta(seconds=(time.time() - start_time))

# evaluate the network
print('[INFO] evaluating network...')

predictions = model.predict(test_x, batch_size=32)

print('[INFO] generating metrics...')

file_title = create_file_title('UNetSeg', hyperparameters)

acc = model.evaluate(test_x, test_y)
output = str(model.metrics_names) + '\n'
output += str(acc) + '\n'
output += 'IOU: {}\n'.format(iou_coef(test_y, predictions))
output += 'Dice: {}\n'.format(dice_coef(test_y, predictions))
output += 'Time taken: {}\n'.format(time_taken)

show_predictions(model, test_x, test_y, 2, output_dir + 'segmentation/' + file_title + '_pred2.png')
show_predictions(model, test_x, test_y, 12, output_dir + 'segmentation/' + file_title + '_pred12.png')
show_predictions(model, test_x, test_y, 22, output_dir + 'segmentation/' + file_title + '_pred22.png')

with open(output_dir + file_title + '_metrics.txt', 'w+') as text_file:
    text_file.write(output)

generate_heatmap(model, test_x, 10, 0, hyperparameters)

print('[END] Finishing script...\n')
