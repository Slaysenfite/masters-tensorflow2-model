import sys

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.utils.np_utils import to_categorical

from configurations.DataSet import binary_ddsm_data_set as data_set
from configurations.TrainingConfig import create_required_directories, IMAGE_DIMS, pso_hyperparameters, create_callbacks
from metrics.MetricsReporter import MetricReporter
from networks.NetworkHelper import compile_with_regularization, create_classification_layers
from training_loops.PsoTrainingLoop import training_loop
from utils.Emailer import results_dispatch
from utils.ImageLoader import load_rgb_images
from utils.ScriptHelper import read_cmd_line_args, generate_script_report

print('Python version: {}'.format(sys.version))
print('Tensorflow version: {}\n'.format(tf.__version__))
print('[BEGIN] Start script...\n')
read_cmd_line_args(data_set, pso_hyperparameters, IMAGE_DIMS)
print(' Image dimensions: {}\n'.format(IMAGE_DIMS))
print(pso_hyperparameters.report_hyperparameters())

print('[INFO] Creating required directories...')
create_required_directories()

# initialize the data and labels
data = []
labels = []

print('[INFO] Loading images...')
data, labels = load_rgb_images(data, labels, data_set, IMAGE_DIMS)

# partition the data into training and testing splits using 70% of
# the data for training and the remaining 30% for testing
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.3, train_size=0.7, random_state=42)

if data_set.is_multiclass:
    print('[INFO] Configure for multiclass classification')
    lb = LabelBinarizer()
    train_y = lb.fit_transform(train_y)
    test_y = lb.transform(test_y)
    loss = 'categorical_crossentropy'
else:
    print('[INFO] Configure for binary classification')
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    loss = 'binary_crossentropy'

base_model = ResNet50(include_top=False,
                      weights=None,
                      input_tensor=None,
                      input_shape=IMAGE_DIMS,
                      pooling=None,
                      classes=2)
model = create_classification_layers(base_model, classes=len(data_set.class_names),
                                     dropout_prob=pso_hyperparameters.dropout)

opt = SGD(lr=pso_hyperparameters.init_lr, decay=pso_hyperparameters.init_lr / pso_hyperparameters.epochs)

compile_with_regularization(model, loss=loss, optimizer=opt, metrics=['accuracy'],
                            regularization_type='l2', attrs=['weight_regularizer'], l2=0.005)

print('[INFO] Adding callbacks')
callbacks = create_callbacks()

H = training_loop(model, pso_hyperparameters, train_x, train_y, test_x, test_y, callbacks)

# evaluate the network
print('[INFO] evaluating network...')

predictions = model.predict(test_x, batch_size=32)

print('[INFO] generating metrics...')

generate_script_report(H, test_y, predictions, data_set, pso_hyperparameters, 'testnet-pso')

reporter = MetricReporter(data_set.name, 'testnet-pso')
cm1 = confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
reporter.plot_confusion_matrix(cm1, classes=data_set.class_names,
                               title='Confusion matrix, without normalization')

reporter.plot_roc(data_set.class_names, test_y, predictions)

reporter.plot_network_metrics(H, 'testnet-pso')

# print('[INFO] serializing network and label binarizer...')
#
# reporter.save_model_to_file(model, lb)

print('[INFO] emailing result...')

results_dispatch(data_set.name, 'testnet-pso')

print('[END] Finishing script...\n')
