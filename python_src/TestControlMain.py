import sys

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from configurations.DataSet import cbis_ddsm_five_data_set as data_set
from configurations.TrainingConfig import IMAGE_DIMS, create_required_directories, hyperparameters, create_callbacks
from metrics.MetricsReporter import MetricReporter
from networks.NetworkHelper import create_classification_layers
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
(train_x, test_x, train_y, test_y) = data_set.split_data_set(data, labels)

print('[INFO] Augmenting data set')
aug = ImageDataGenerator(
    horizontal_flip=True,
    fill_mode="nearest")

print("[INFO] Training data shape: " + str(train_x.shape))
print("[INFO] Training label shape: " + str(train_y.shape))

loss, train_y, test_y = data_set.get_dataset_labels(train_y, test_y)

base_model = ResNet50(include_top=False,
                 weights=None,
                 input_tensor=None,
                 input_shape=IMAGE_DIMS,
                 pooling=None,
                 classes=2)
model = create_classification_layers(base_model, classes=len(data_set.class_names), dropout_prob=hyperparameters.dropout)

opt = Adam(learning_rate=hyperparameters.init_lr, decay=True)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.Precision(),
                                                                       tf.keras.metrics.Recall(),
                                                                       tf.keras.metrics.Accuracy(),
                                                                       tf.keras.metrics.BinaryAccuracy()])

print('[INFO] Adding callbacks')
callbacks = create_callbacks()

# train the network
H = model.fit(x=aug.flow(train_x, train_y, batch_size=hyperparameters.batch_size), validation_data=(test_x, test_y),
              steps_per_epoch=len(train_x) // hyperparameters.batch_size, epochs=hyperparameters.epochs,
              callbacks=callbacks)

# evaluate the network
print('[INFO] evaluating network...')

predictions = model.predict(test_x, batch_size=hyperparameters.batch_size)

print('[INFO] generating metrics...')

generate_script_report(H, test_y, predictions, data_set, hyperparameters, 'testnet-control')

reporter = MetricReporter(data_set.name, 'testnet-control')
cm1 = confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
reporter.plot_confusion_matrix(cm1, classes=data_set.class_names,
                               title='Confusion matrix, without normalization')

reporter.plot_roc(data_set.class_names, test_y, predictions)

reporter.plot_network_metrics(H, 'testnet-control')

print('[END] Finishing script...\n')
