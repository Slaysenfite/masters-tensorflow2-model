import smtplib
import sys

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from configurations.GlobalConstants import IMAGE_DIMS
from metrics.MetricsReporter import plot_confusion_matrix,plot_network_metrics, plot_roc, \
    save_model_to_file
from model.DataSet import ddsm_data_set
from model.Hyperparameters import hyperparameters
from networks.VggNet16 import SmallVGGNet
from utils.Emailer import results_dispatch
from utils.ImageLoader import load_images
from utils.ScriptHelper import generate_script_report

print('Python version: {}'.format(sys.version))
print('Tensorflow version: {}\n'.format(tf.__version__))
print('[BEGIN] Start script...\n')
print(' Image dimensions: {}\n'.format(IMAGE_DIMS))
print(hyperparameters.report_hyperparameters())

# initialize the data and labels
data = []
labels = []

print('[INFO] Loading images...')
data, labels = load_images(data, labels, ddsm_data_set, IMAGE_DIMS)

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

model = SmallVGGNet.build(IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2], classes=len(lb.classes_))

print('[INFO] Model summary...')
model.summary()

opt = SGD(lr=hyperparameters.init_lr, decay=hyperparameters.init_lr / hyperparameters.epochs)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# train the network
H = model.fit_generator(aug.flow(train_x, train_y, batch_size=hyperparameters.batch_size),
                        validation_data=(test_x, test_y), steps_per_epoch=len(train_x) // hyperparameters.batch_size,
                        epochs=hyperparameters.epochs)

# evaluate the network
print('[INFO] evaluating network...')

predictions = model.predict(test_x, batch_size=32)

print('[INFO] generating metrics...')

generate_script_report(H, test_y, predictions, ddsm_data_set, hyperparameters)

cm1 = confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
plot_confusion_matrix(cm1, classes=ddsm_data_set.class_names,
                      title='Confusion matrix, without normalization')

plot_roc(ddsm_data_set.class_names, test_y, predictions)

plot_network_metrics(hyperparameters.epochs, H, "VggNet")

print('[INFO] serializing network and label binarizer...')

save_model_to_file(model, lb)

print('[INFO] emailing result...')

try:
    results_dispatch('ddsm', "vggnet")
except smtplib.SMTPAuthenticationError:
    print('[ERROR] Email credentials could not be authenticated')

print('[END] Finishing script...\n')
