import sys
import time
from datetime import timedelta

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from configurations.DataSet import bcs_data_set as data_set
from configurations.TrainingConfig import IMAGE_DIMS, create_required_directories, hyperparameters, create_callbacks, \
    MODEL_OUTPUT
from metrics.MetricsReporter import MetricReporter
from networks.NetworkHelper import create_classification_layers, compile_with_regularization
from training_loops.CustomCallbacks import RunMetaHeuristicOnPlateau
from training_loops.CustomTrainingLoop import training_loop
from utils.ImageLoader import load_rgb_images, supplement_training_data
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

# initialize the data and labels
data = []
labels = []

print('[INFO] Loading images...')
data, labels = load_rgb_images(data, labels, data_set, IMAGE_DIMS)

# partition the data into training and testing splits using 70% of
# the data for training and the remaining 30% for testing
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.2, train_size=0.8, random_state=42)


if hyperparameters.augmentation:
    print('[INFO] Augmenting data set')
    aug = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=10,
        zoom_range=0.05,
        fill_mode='nearest')

    train_x, train_y = supplement_training_data(aug, train_x, train_y, multiclass=False)

print('[INFO] Training data shape: ' + str(train_x.shape))
print('[INFO] Training label shape: ' + str(train_y.shape))

loss, train_y, test_y = data_set.get_dataset_labels(train_y, test_y)

if hyperparameters.preloaded_weights:
    print('[INFO] Loading imagenet weights')
    weights = 'imagenet'
else:
    weights = None
model = Xception(
        include_top=False,
        weights=weights,
        input_shape=IMAGE_DIMS,
        classes=len(data_set.class_names))
model = create_classification_layers(base_model=model, classes=len(data_set.class_names))

if hyperparameters.weights_of_experiment_id is not None:
    path_to_weights = '{}{}.h5'.format(MODEL_OUTPUT, hyperparameters.weights_of_experiment_id)
    print('[INFO] Loading weights from {}'.format(path_to_weights))
    model.load_weights(path_to_weights)

# Compile model
compile_with_regularization(model=model,
                            loss='binary_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy'],
                            regularization_type='l2')

# Setup callbacks
callbacks = create_callbacks(hyperparameters)

if hyperparameters.meta_heuristic != 'none':
    meta_callback = RunMetaHeuristicOnPlateau(
        X=train_x, y=train_y, meta_heuristic=hyperparameters.meta_heuristic, population_size=25, iterations=10,
monitor='val_loss', factor=0.2, patience=4, verbose=1, mode='min',
        min_delta=0.05, cooldown=0)
    callbacks.append(meta_callback)

# train the network
start_time = time.time()

if hyperparameters.tf_fit:
    H = model.fit(train_x, train_y, batch_size=hyperparameters.batch_size, validation_data=(test_x, test_y),
                  steps_per_epoch=len(train_x) // hyperparameters.batch_size, epochs=hyperparameters.epochs,
                  callbacks=callbacks)
else:
    H = training_loop(model, opt, hyperparameters, train_x, train_y, test_x, test_y,
                      meta_heuristic=hyperparameters.meta_heuristic,
                      meta_heuristic_order=hyperparameters.meta_heuristic_order)

time_taken = timedelta(seconds=(time.time() - start_time))

# evaluate the network
print('[INFO] evaluating network...')

acc = model.evaluate(test_x, test_y)
print(str(model.metrics_names))
print(str(acc))

predictions = model.predict(test_x)

print('[INFO] generating metrics...')

file_title = create_file_title('Xception', hyperparameters)

model.save(filepath=MODEL_OUTPUT + file_title + '.h5', save_format='h5')

generate_script_report(H, model, test_x, test_y, predictions, time_taken, data_set, hyperparameters, file_title)

reporter = MetricReporter(data_set.name,file_title)
cm1 = confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
reporter.plot_confusion_matrix(cm1, classes=data_set.class_names,
                               title='Confusion matrix, without normalization')

reporter.plot_roc(data_set.class_names, test_y, predictions)

reporter.plot_network_metrics(H, file_title)

print('[END] Finishing script...\n')
