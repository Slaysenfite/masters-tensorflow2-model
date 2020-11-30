import sys

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam

from configurations.DataSet import binary_ddsm_data_set as data_set
from configurations.TrainingConfig import create_required_directories, IMAGE_DIMS
from configurations.TrainingConfig import hybrid_hyperparameters as hyperparameters
from metrics.MetricsReporter import MetricReporter
from networks.NetworkHelper import compile_with_regularization, create_classification_layers
from training_loops.CustomTrainingLoop import training_loop
from utils.Emailer import results_dispatch
from utils.ImageLoader import load_rgb_images
from utils.ScriptHelper import read_cmd_line_args, generate_script_report

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
(train_x, test_x, train_y, test_y) = data_set.split_data_set(data, labels)

loss, train_y, test_y = data_set.get_dataset_labels(train_y, test_y)

base_model = ResNet50(include_top=False,
                      weights=None,
                      input_tensor=None,
                      input_shape=IMAGE_DIMS,
                      pooling=None,
                      classes=2)
model = create_classification_layers(base_model, classes=len(data_set.class_names),
                                     dropout_prob=hyperparameters.dropout)

opt = Adam(learning_rate=hyperparameters.init_lr, decay=True)

compile_with_regularization(model, loss=loss, optimizer=opt, metrics=['accuracy'],
                            regularization_type='l2', attrs=['weight_regularizer'], l2=0.005)

H = training_loop(model, opt, hyperparameters, train_x, train_y, test_x, test_y, pso_layer=(Conv2D, Dense),
                  gd_layer=(Conv2D, Dense))

# evaluate the network
print('[INFO] evaluating network...')

predictions = model.predict(test_x, batch_size=32)

print('[INFO] generating metrics...')

generate_script_report(H, test_y, predictions, data_set, hyperparameters, 'testnet-hybrid')

reporter = MetricReporter(data_set.name, 'testnet-hybrid')
cm1 = confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
reporter.plot_confusion_matrix(cm1, classes=data_set.class_names,
                               title='Confusion matrix, without normalization')

reporter.plot_roc(data_set.class_names, test_y, predictions)

reporter.plot_network_metrics(H, 'testnet-hybrid')

print('[INFO] emailing result...')

results_dispatch(data_set.name, 'testnet-hybrid')

print('[END] Finishing script...\n')
