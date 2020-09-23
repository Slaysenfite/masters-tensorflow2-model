import sys

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.applications.resnet50 import resnet50
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from configurations.GConstants import IMAGE_DIMS, create_required_directories
from metrics.MetricsReporter import MetricReporter
from model.DataSet import ddsm_data_set as data_set
from model.Hyperparameters import hyperparameters, create_callbacks
from utils.Emailer import results_dispatch
from utils.ImageLoader import load_rgb_images
from utils.ScriptHelper import generate_script_report, read_cmd_line_args

print('Python version: {}'.format(sys.version))
print('Tensorflow version: {}\n'.format(tf.__version__))
print('[BEGIN] Start script...\n')
read_cmd_line_args(data_set, hyperparameters, IMAGE_DIMS)
print(' Image dimensions: {}\n'.format(IMAGE_DIMS))
print(hyperparameters.report_hyperparameters())

print('[INFO] Creating required directories...')
create_required_directories()

# initialize the data and labels
data = []
labels = []

print('[INFO] Loading images...')
data, labels = load_rgb_images(data, labels, data_set, IMAGE_DIMS)

# partition the data into training and testing splits using 70% of
# the data for training and the remaining 30% for testing
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.2, train_size=0.8, random_state=42)

'[INFO] Augmenting data set'

aug = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=10,
    zoom_range=0.05,
    fill_mode="nearest")

# train_x, train_y = supplement_training_data(aug, train_x, train_y)

print("[INFO] Training data shape: " + str(train_x.shape))
print("[INFO] Training label shape: " + str(train_y.shape))

# binarize the class labels
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

model = resnet50.ResNet50(include_top=True,
                 weights=None,
                 input_tensor=None,
                 input_shape=IMAGE_DIMS,
                 pooling=None,
                 classes=3)

opt = Adam(learning_rate=hyperparameters.init_lr, decay=True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# Setup callbacks
callbacks = create_callbacks()

# train the network
H = model.fit(x=aug.flow(train_x, train_y, batch_size=hyperparameters.batch_size), validation_data=(test_x, test_y),
              steps_per_epoch=len(train_x) // hyperparameters.batch_size, epochs=hyperparameters.epochs,
              callbacks=callbacks)

# evaluate the network
print('[INFO] evaluating network...')

predictions = model.predict(test_x, batch_size=hyperparameters.batch_size)

print('[INFO] generating metrics...')

generate_script_report(H, test_y, predictions, data_set, hyperparameters, 'resnet')

reporter = MetricReporter(data_set.name, 'resnet')
cm1 = confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
reporter.plot_confusion_matrix(cm1, classes=data_set.class_names,
                               title='Confusion matrix, without normalization')

reporter.plot_roc(data_set.class_names, test_y, predictions)

reporter.plot_network_metrics(H, 'ResNet')

# print('[INFO] serializing network and label binarizer...')
#
# reporter.save_model_to_file(model, lb)

print('[INFO] emailing result...')

results_dispatch(data_set.name, 'resnet')

print('[END] Finishing script...\n')
