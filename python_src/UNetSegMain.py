import sys
import time
from datetime import timedelta

import tensorflow as tf
from IPython.core.display import clear_output
from numpy import expand_dims
from numpy.ma import array
from scipy.spatial.distance import dice
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizer_v2.adam import Adam

from configurations.DataSet import cbis_seg_data_set as data_set
from configurations.TrainingConfig import IMAGE_DIMS, hyperparameters, output_dir
from networks.UNetSeg import unet_seg
from training_loops.CustomTrainingLoop import training_loop
from utils.ImageLoader import load_seg_images
from utils.ScriptHelper import create_file_title, read_cmd_line_args

print('Python version: {}'.format(sys.version))
print('Tensorflow version: {}\n'.format(tf.__version__))
print('[BEGIN] Start script...\n')
hyperparameters, opt, data_set = read_cmd_line_args(hyperparameters, data_set)
print(' Image dimensions: {}\n'.format(IMAGE_DIMS))
print(hyperparameters.report_hyperparameters())

print('[INFO] Loading images...')
roi_data, roi_labels = load_seg_images(data_set, path_suffix='roi', image_dimensions=IMAGE_DIMS)
data, labels = load_seg_images(data_set, image_dimensions=IMAGE_DIMS)
(train_x, test_x, train_y, test_y) = train_test_split(data, roi_data, test_size=0.3, train_size=0.7, random_state=42)

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

clear_output()

opt = Adam()

model = unet_seg(IMAGE_DIMS)
model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

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
H = training_loop(model, opt, hyperparameters, train_x, train_y, test_x, test_y,
                  meta_heuristic=hyperparameters.meta_heuristic,
                  meta_heuristic_order=hyperparameters.meta_heuristic_order)
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

def iou_coef(y_true, y_pred, smooth=1):
    m = tf.keras.metrics.MeanIoU(num_classes=3)
    m.update_state(y_true, y_pred)
    return m.result().numpy()

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = array(K.flatten(y_true))
    y_pred_f = array(K.flatten(y_pred))
    return dice(y_true_f, y_pred_f)


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
