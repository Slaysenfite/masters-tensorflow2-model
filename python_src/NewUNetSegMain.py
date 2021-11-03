import gc
import sys
import time
from datetime import timedelta

import tensorflow as tf
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.metrics import MeanIoU

from configurations.DataSet import cbis_seg_data_set as data_set
from configurations.TrainingConfig import IMAGE_DIMS, hyperparameters, output_dir, MODEL_OUTPUT, \
    create_required_directories
from metrics.MetricsUtil import iou_coef, dice_coef
from networks.NetworkHelper import generate_heatmap
from networks.UNet import build_pretrained_unet
from networks.UNetSeg import unet_seg
from training_loops.CustomTrainingLoop import training_loop
from training_loops.OptimizerHelper import calc_seg_fitness
from utils.ImageLoader import load_seg_images
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

print('[INFO] Training data shape: ' + str(train_x.shape))
print('[INFO] Training label shape: ' + str(train_y.shape))

if hyperparameters.preloaded_weights:
    model = build_pretrained_unet(IMAGE_DIMS, len(data_set.class_names))
else:
    model = unet_seg(IMAGE_DIMS)

if hyperparameters.weights_of_experiment_id is not None:
    path_to_weights = '{}{}.h5'.format(MODEL_OUTPUT, hyperparameters.weights_of_experiment_id)
    print('[INFO] Loading weights from {}'.format(path_to_weights))
    model.load_weights(path_to_weights)

# Compile model
model.compile(loss=BinaryCrossentropy(),
              optimizer=opt,
              metrics=['accuracy', MeanIoU(num_classes=len(data_set.class_names))], )

start_time = time.time()

print('META-HEURISTIC')

H = training_loop(model, hyperparameters, train_x, train_y, test_x, test_y, fitness_function=calc_seg_fitness,
                  meta_heuristic=hyperparameters.meta_heuristic, num_solutions=10, iterations=5)
time_taken = timedelta(seconds=(time.time() - start_time))
print(time_taken)

print('EVALUATION')

generate_heatmap(model, test_x, 10, 0, hyperparameters, '_meta_pass')
generate_heatmap(model, test_x, 10, 1, hyperparameters, '_meta_pass')

predictions = model.predict(test_x, batch_size=32)

show_predictions(model, test_x, test_y, 2, output_dir + 'segmentation/' + hyperparameters.experiment_id + '_pred2.png')
show_predictions(model, test_x, test_y, 12,
                 output_dir + 'segmentation/' + hyperparameters.experiment_id + '_pred12.png')
show_predictions(model, test_x, test_y, 22,
                 output_dir + 'segmentation/' + hyperparameters.experiment_id + '_pred22.png')

# evaluate the network
file_title = create_file_title('UNetSeg2Meta', hyperparameters)

acc = model.evaluate(test_x, test_y)
output = str(model.metrics_names) + '\n'
output += str(acc) + '\n'
output += 'IOU: {}\n'.format(iou_coef(test_y, predictions))
output += 'Dice: {}\n'.format(dice_coef(test_y, predictions))

with open(output_dir + file_title + '_metrics.txt', 'w+') as text_file:
    text_file.write(output)

print('[END] Finishing script...\n')
