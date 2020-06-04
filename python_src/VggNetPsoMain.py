import sys

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from configurations.GConstants import IMAGE_DIMS
from metrics.MetricsReporter import MetricReporter
from model.DataSet import mias_data_set as data_set
from model.Hyperparameters import hyperparameters
from networks.VggNet19 import Vgg19Net
from psokeras import Optimizer
from utils.Emailer import results_dispatch
from utils.ImageLoader import load_rgb_images
from utils.ScriptHelper import generate_script_report, read_cmd_line_args

print('Python version: {}'.format(sys.version))
print('Tensorflow version: {}\n'.format(tf.__version__))
print('[BEGIN] Start script...\n')
read_cmd_line_args(data_set, hyperparameters, IMAGE_DIMS)
print(' Image dimensions: {}\n'.format(IMAGE_DIMS))
print(hyperparameters.report_hyperparameters())

# initialize the data and labels
data = []
labels = []

print('[INFO] Loading images...')
data, labels = load_rgb_images(data, labels, data_set, IMAGE_DIMS)

# partition the data into training and testing splits using 70% of
# the data for training and the remaining 30% for testing
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.3, train_size=0.7, random_state=42)

# binarize the class labels
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# construct the image generator for data augmentation
print('[INFO] Augmenting data set')
aug = ImageDataGenerator()


model = Vgg19Net.build(IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2], classes=len(lb.classes_))

pso = Optimizer(model,
                loss='mse',
                n=30,  # Number of particles
                acceleration=1.0,  # Contribution of recursive particle velocity (acceleration)
                local_rate=0.6,  # Contribution of locally best weights to new velocity
                global_rate=0.4)  # Contribution of globally best weights to new velocity

pso.fit(train_x, test_x, steps=300, batch_size=hyperparameters.batch_size)

# Get a copy of the model with the globally best weights
model_p = pso.get_best_model()

# model.compile(loss='mse', optimizer=pso, metrics=['accuracy'])


#
# # train the network
# H = model.fit_generator(aug.flow(train_x, train_y, batch_size=hyperparameters.batch_size),
#                         validation_data=(test_x, test_y), steps_per_epoch=len(train_x) // hyperparameters.batch_size,
#                         epochs=hyperparameters.epochs)
#
# # evaluate the network
# print('[INFO] evaluating network...')
#
# predictions = model.predict(test_x, batch_size=32)
#
# print('[INFO] generating metrics...')
#
# generate_script_report(H, test_y, predictions, data_set, hyperparameters)
#
# reporter = MetricReporter(data_set.name, 'vggpython.net')
# cm1 = confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
# reporter.plot_confusion_matrix(cm1, classes=data_set.class_names,
#                                title='Confusion matrix, without normalization')
#
# reporter.plot_roc(data_set.class_names, test_y, predictions)
#
# reporter.plot_network_metrics(hyperparameters.epochs, H, "VggNet")
#
# print('[INFO] serializing network and label binarizer...')
#
# reporter.save_model_to_file(model, lb)
#
# reporter.print('[INFO] emailing result...')
#
# results_dispatch(data_set.name, "vggnet")
#
# print('[END] Finishing script...\n')