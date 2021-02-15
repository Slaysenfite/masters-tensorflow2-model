import sys

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from configurations.DataSet import cbis_ddsm_data_set as data_set
from configurations.TrainingConfig import IMAGE_DIMS, create_required_directories, hyperparameters, create_callbacks
from metrics.MetricsReporter import MetricReporter
from networks.ResNet import ResnetBuilder
from training_loops.CustomTrainingLoop import training_loop
from utils.Emailer import results_dispatch
from utils.ImageLoader import load_rgb_images, supplement_training_data
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

train_x, train_y = supplement_training_data(aug, train_x, train_y)

print("[INFO] Training data shape: " + str(train_x.shape))
print("[INFO] Training label shape: " + str(train_y.shape))

loss, train_y, test_y = data_set.get_dataset_labels(train_y, test_y)

model = ResnetBuilder.build_resnet_50(IMAGE_DIMS, len(data_set.class_names))

opt = Adam(learning_rate=hyperparameters.init_lr, decay=True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', Precision(), Recall()])


# Setup callbacks
callbacks = create_callbacks()

H = training_loop(model, opt, hyperparameters, train_x, train_y, test_x, test_y, metaheuristic='ga',
                  metaheuristic_order='first')

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

print('[END] Finishing script...\n')
