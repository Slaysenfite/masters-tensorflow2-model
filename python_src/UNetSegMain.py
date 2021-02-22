import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import re
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ast
from PIL import Image, ImageDraw
from configurations.DataSet import cbis_ddsm_data_set as data_set

from configurations.TrainingConfig import hyperparameters, IMAGE_DIMS, create_required_directories
from networks import UNet
from utils.ImageLoader import load_greyscale_images

from utils.ScriptHelper import read_cmd_line_args

print('Python version: {}'.format(sys.version))
print('Tensorflow version: {}\n'.format(tf.__version__))
print('[BEGIN] Start script...\n')
hyperparameters, opt = read_cmd_line_args(hyperparameters)
print(' Image dimensions: {}\n'.format(IMAGE_DIMS))
print(hyperparameters.report_hyperparameters())

print('[INFO] Creating required directories...')
create_required_directories()

# initialize the data and labels
data = []
labels = []

print('[INFO] Loading images...')
data, labels = load_greyscale_images(data, labels, data_set, [IMAGE_DIMS[0], IMAGE_DIMS[1], 1])

plt.subplots()
def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

model = UNet.build([IMAGE_DIMS[0], IMAGE_DIMS[1], 1], len(data_set.class_names))

