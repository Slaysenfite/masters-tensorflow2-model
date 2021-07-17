import sys
import time
from datetime import timedelta

import tensorflow as tf
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from configurations.DataSet import cbis_ddsm_data_set as data_set
from configurations.TrainingConfig import IMAGE_DIMS, create_required_directories
from configurations.TrainingConfig import hyperparameters
from metrics.MetricsReporter import MetricReporter
from networks.UNet import build_unet
from training_loops.CustomTrainingLoop import training_loop
from utils.ImageLoader import load_greyscale_images, show_examples, supplement_training_data
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
data, labels = load_greyscale_images(data, labels, data_set, [IMAGE_DIMS[0], IMAGE_DIMS[1], 1])

# partition the data into training and testing splits using 70% of
# the data for training and the remaining 30% for testing
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.2, train_size=0.8, random_state=42)
# show_examples('CBIS-DDSM Example Images', train_x, test_x, train_y, test_y, items=9)

if hyperparameters.augmentation:
    print('[INFO] Augmenting data set')
    aug = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=10,
        zoom_range=0.05,
        fill_mode="nearest")

    train_x, train_y = supplement_training_data(aug, train_x, train_y)

print('[INFO] Training data shape: ' + str(train_x.shape))
print('[INFO] Training label shape: ' + str(train_y.shape))

# # plot first few images
# for i in range(9):
#     # define subplot
#     pyplot.subplot(330 + 1 + i)
#     # plot raw pixel data
#     pyplot.imshow(train_x[i], cmap=pyplot.get_cmap('gray'))
# # show the figure
# pyplot.show()

loss, train_y, test_y = data_set.get_dataset_labels(train_y, test_y)

model = build_unet([IMAGE_DIMS[0], IMAGE_DIMS[1], 1], len(data_set.class_names))

opt = Adam(learning_rate=hyperparameters.init_lr)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', Precision(), Recall()])

start_time = time.time()
H = training_loop(model, opt, hyperparameters, train_x, train_y, test_x, test_y,
                  meta_heuristic=hyperparameters.meta_heuristic,
                  meta_heuristic_order=hyperparameters.meta_heuristic_order)
time_taken = timedelta(seconds=(time.time() - start_time))

# H =model.fit(x=aug.flow(train_x, train_y, batch_size=hyperparameters.batch_size), validation_data=(test_x, test_y),
#               steps_per_epoch=len(train_x) // hyperparameters.batch_size, epochs=hyperparameters.epochs,
#               callbacks=callbacks)

# evaluate the network
print('[INFO] evaluating network...')

acc = model.evaluate(test_x, test_y)
print(str(model.metrics_names))
print(str(acc))

predictions = model.predict(test_x, batch_size=32)

print('[INFO] generating metrics...')

file_title = create_file_title('UNet', hyperparameters)

generate_script_report(H, model, test_x, test_y, predictions, time_taken, data_set, hyperparameters, file_title)

reporter = MetricReporter(data_set.name, file_title)
cm1 = confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
reporter.plot_confusion_matrix(cm1, classes=data_set.class_names,
                               title='Confusion matrix, without normalization')

reporter.plot_roc(data_set.class_names, test_y, predictions)

reporter.plot_network_metrics(H, file_title)

print('[END] Finishing script...\n')
