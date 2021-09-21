import gc
import sys
import time
from datetime import timedelta

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from numpy import AxisError
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.metrics import Precision, Recall, AUC

from configurations.DataSet import cbis_ddsm_data_set as data_set
from configurations.DataSet import cbis_seg_data_set as c_data_set
from configurations.TrainingConfig import IMAGE_DIMS, create_required_directories, hyperparameters, create_callbacks, \
    MODEL_OUTPUT
from metrics.MetricsReporter import MetricReporter
from networks.NetworkHelper import create_classification_layers, compile_with_regularization, generate_heatmap
from training_loops.CustomCallbacks import RunMetaHeuristicOnPlateau
from utils.ImageLoader import load_rgb_images, load_seg_images, supplement_training_data
from utils.ScriptHelper import generate_script_report, read_cmd_line_args, create_file_title

print('Python version: {}'.format(sys.version))
print('Tensorflow version: {}\n'.format(tf.__version__))
print('[BEGIN] Start script...\n')
hyperparameters, opt, data_set = read_cmd_line_args(hyperparameters, data_set)
print(' Dataset: {}\n'.format(data_set.name))
print(' Image dimensions: {}\n'.format(IMAGE_DIMS))
print(hyperparameters.report_hyperparameters())

print('[INFO] Creating required directories...')
create_required_directories()
gc.enable()

"""FIRST PASS"""

print('[INFO] Loading cropped images...')
c_train_x, c_train_y = load_seg_images(c_data_set, path_suffix='cropped', image_dimensions=IMAGE_DIMS,
                                       subset='Training')
c_test_x, c_test_y = load_seg_images(c_data_set, path_suffix='cropped', image_dimensions=IMAGE_DIMS, subset='Test')

if hyperparameters.augmentation:
    print('[INFO] Augmenting data set')
    aug = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=10,
        zoom_range=0.05,
        fill_mode='nearest')

    train_x, train_y = supplement_training_data(aug, c_train_x, c_train_y, multiclass=False)

    print('[INFO] Training data shape: ' + str(train_x.shape))

loss, c_train_y, c_test_y = c_data_set.get_dataset_labels(c_train_y, c_test_y)

if hyperparameters.preloaded_weights:
    print('[INFO] Loading imagenet weights')
    weights = 'imagenet'
else:
    weights = None
model = ResNet50V2(
    include_top=False,
    weights=weights,
    input_shape=IMAGE_DIMS,
    classes=c_data_set.get_num_classes())
model = create_classification_layers(base_model=model,
                                     classes=c_data_set.get_num_classes(),
                                     dropout_prob=hyperparameters.dropout_prob)

# Compile model
compile_with_regularization(model=model,
                            loss='binary_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy', Precision(), Recall(), AUC()],
                            regularization_type='l2',
                            l2=hyperparameters.l2)

# Setup callbacks
callbacks = create_callbacks(hyperparameters)

# train the network
start_time = time.time()
H = model.fit(c_train_x, c_train_y, batch_size=hyperparameters.batch_size, validation_data=(c_test_x, c_test_y),
              steps_per_epoch=len(c_train_x) // hyperparameters.batch_size, epochs=hyperparameters.epochs,
              callbacks=callbacks)

model = None
c_train_x = None
c_train_y = None
c_test_x = None
c_test_y = None
gc.collect()

"""SECOND PASS"""

print('[INFO] Loading images...')
test_x, test_y = load_rgb_images(data_set, IMAGE_DIMS, subset='Test')
train_x, train_y = load_rgb_images(data_set, IMAGE_DIMS, subset='Training')
loss, train_y, test_y = data_set.get_dataset_labels(train_y, test_y)

model = ResNet50V2(
    include_top=False,
    weights=None,
    input_shape=IMAGE_DIMS,
    classes=data_set.get_num_classes())
model = create_classification_layers(base_model=model,
                                     classes=data_set.get_num_classes(),
                                     dropout_prob=hyperparameters.dropout_prob)

# Compile model
compile_with_regularization(model=model,
                            loss='binary_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy', Precision(), Recall(), AUC()],
                            regularization_type='l2',
                            l2=hyperparameters.l2)

path_to_weights = '{}{}.h5'.format(MODEL_OUTPUT, hyperparameters.experiment_id)
print('[INFO] Loading weights from {}'.format(path_to_weights))
model.load_weights(path_to_weights)

if hyperparameters.meta_heuristic != 'none':
    meta_callback = RunMetaHeuristicOnPlateau(
        X=train_x, y=train_y, meta_heuristic=hyperparameters.meta_heuristic, population_size=30, iterations=10,
        monitor='val_loss', patience=6, verbose=1, mode='min', min_delta=0.001, cooldown=4)
    callbacks.append(meta_callback)

H = model.fit(train_x, train_y, batch_size=hyperparameters.batch_size, validation_data=(test_x, test_y),
              steps_per_epoch=len(train_x) // hyperparameters.batch_size, epochs=hyperparameters.epochs,
              callbacks=callbacks)
time_taken = timedelta(seconds=(time.time() - start_time))

"""EVALUATION"""

# evaluate the network
print('[INFO] evaluating network...')

acc = model.evaluate(test_x, test_y)
print(str(model.metrics_names))
print(str(acc))

predictions = model.predict(test_x)

print('[INFO] generating metrics...')

file_title = create_file_title('ResNet', hyperparameters)

generate_heatmap(model, test_x, 10, 0, hyperparameters)
generate_heatmap(model, test_x, 10, 1, hyperparameters)

generate_script_report(H, model, test_x, test_y, predictions, time_taken, data_set, hyperparameters, file_title)

reporter = MetricReporter(data_set.name, file_title)
try:
    cm1 = confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
    reporter.plot_confusion_matrix(cm1, classes=data_set.class_names,
                               title='Confusion matrix, without normalization')
except AxisError:
    print('pee pee poo poo')

reporter.plot_roc(data_set.class_names, test_y, predictions)

reporter.plot_network_metrics(H, file_title)



model.save(filepath=MODEL_OUTPUT + file_title + '2.h5', save_format='h5', overwrite=True)

print('[END] Finishing script...\n')
