import sys
import time
from datetime import timedelta

import tensorflow as tf
from IPython.core.display import clear_output
from numpy import expand_dims
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizer_v2.adam import Adam

from configurations.DataSet import cbis_seg_data_set as data_set
from configurations.TrainingConfig import IMAGE_DIMS, hyperparameters, output_dir, create_callbacks, MODEL_OUTPUT
from metrics.MetricsUtil import iou_coef, dice_coef
from networks.UNet import build_pretrained_unet
from networks.UNetSeg import unet_seg
from training_loops.CustomCallbacks import RunMetaHeuristicOnPlateau
from training_loops.CustomTrainingLoop import training_loop
from training_loops.OptimizerHelper import calc_seg_fitness
from utils.ImageLoader import load_seg_images
from utils.ScriptHelper import create_file_title, read_cmd_line_args

print('Python version: {}'.format(sys.version))
print('Tensorflow version: {}\n'.format(tf.__version__))
print('[BEGIN] Start script...\n')
hyperparameters, opt, data_set = read_cmd_line_args(hyperparameters, data_set)
print(' Image dimensions: {}\n'.format(IMAGE_DIMS))
print(hyperparameters.report_hyperparameters())

print('[INFO] Loading images...')
roi_data, roi_labels = load_seg_images(data_set, path_suffix='roi', image_dimensions=(IMAGE_DIMS[0], IMAGE_DIMS[1], 1))
data, labels = load_seg_images(data_set, image_dimensions=IMAGE_DIMS)

(train_x, test_x, train_y, test_y) = train_test_split(data, roi_data, test_size=0.3, train_size=0.7, random_state=42)

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

clear_output()

opt = Adam()

if hyperparameters.preloaded_weights:
    model = build_pretrained_unet(IMAGE_DIMS, len(data_set.class_names))
else:
    model = unet_seg(IMAGE_DIMS)

if hyperparameters.weights_of_experiment_id is not None:
    path_to_weights = '{}{}.h5'.format(MODEL_OUTPUT, hyperparameters.weights_of_experiment_id)
    print('[INFO] Loading weights from {}'.format(path_to_weights))
    model.load_weights(path_to_weights)

model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Setup callbacks
callbacks = create_callbacks(hyperparameters)

if hyperparameters.meta_heuristic != 'none':
    meta_callback = RunMetaHeuristicOnPlateau(
        X=train_x, y=train_y, meta_heuristic=hyperparameters.meta_heuristic, population_size=25, iterations=10,
monitor='val_loss', factor=0.2, patience=4, verbose=1, mode='min',
        min_delta=0.05, cooldown=0)
    callbacks.append(meta_callback)

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(test_x, index=2, title='pred.png'):
    import matplotlib.pyplot as plt
    image = test_x[index]
    image = expand_dims(image, axis=0)
    pred_mask = model.predict(image)
    display(plt, [test_x[index], test_y[index], pred_mask[0]], title)


start_time = time.time()

if hyperparameters.tf_fit:
     H = model.fit(train_x, train_y, batch_size=hyperparameters.batch_size, validation_data=(test_x, test_y),
                  steps_per_epoch=len(train_x) // hyperparameters.batch_size, epochs=hyperparameters.epochs)
else:
    H = training_loop(model, opt, hyperparameters, train_x, train_y, test_x, test_y,
                      meta_heuristic=hyperparameters.meta_heuristic,
                      meta_heuristic_order=hyperparameters.meta_heuristic_order,
                      fitness_function=calc_seg_fitness, task='segmentation')
time_taken = timedelta(seconds=(time.time() - start_time))


def display(plt, display_list, file_title):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.savefig(file_title)
  plt.clf()
  # plt.show()

show_predictions(test_x, 3)

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


show_predictions(test_x, 2, output_dir + file_title + '_pred2.png')
show_predictions(test_x, 12, output_dir + file_title + '_pred12.png')
show_predictions(test_x, 22, output_dir + file_title + '_pred22.png')

with open(output_dir + file_title + '_metrics.txt', 'w+') as text_file:
    text_file.write(output)

print('[END] Finishing script...\n')
