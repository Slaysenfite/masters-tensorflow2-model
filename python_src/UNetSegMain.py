import sys

import matplotlib.pyplot as plt
import tensorflow as tf

from configurations.DataSet import cbis_seg_data_set as data_set
from configurations.TrainingConfig import hyperparameters, IMAGE_DIMS, create_required_directories
from networks import UNet
from utils.ImageLoader import load_seg_images
from utils.ScriptHelper import read_cmd_line_args

print('Python version: {}'.format(sys.version))
print('Tensorflow version: {}\n'.format(tf.__version__))
print('[BEGIN] Start script...\n')
hyperparameters, opt = read_cmd_line_args(hyperparameters)
print(' Image dimensions: {}\n'.format(IMAGE_DIMS))
print(hyperparameters.report_hyperparameters())

print('[INFO] Creating required directories...')
create_required_directories()

print('[INFO] Loading images...')
c_data, c_labels = load_seg_images(data_set, path_suffix='cropped', image_dimensions=[IMAGE_DIMS[0], IMAGE_DIMS[1], 1])
roi_data, roi_labels = load_seg_images(data_set, path_suffix='roi', image_dimensions=[IMAGE_DIMS[0], IMAGE_DIMS[1], 1])
data, labels = load_seg_images(data_set, [IMAGE_DIMS[0], IMAGE_DIMS[1], 1])

plt.subplots()
def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

model = UNet.build([IMAGE_DIMS[0], IMAGE_DIMS[1], 1], len(data_set.class_names))

