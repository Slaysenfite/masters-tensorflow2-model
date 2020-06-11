import sys

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import GradientTape
from tensorflow.python.data import Dataset
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from model.DataSet import mias_data_set as data_set
from model.Hyperparameters import hyperparameters
from optimizers.OptimizerHelper import VggOneBlock
from utils.ImageLoader import load_rgb_images
from utils.ScriptHelper import read_cmd_line_args

IMAGE_DIMS = (64, 64, 3)

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
(train_x, test_x, train_y, test_y) = train_test_split(data.astype(np.float32), labels, test_size=0.3, train_size=0.7,
                                                      random_state=42)

# binarize the class labels
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

model = VggOneBlock.build(IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2], classes=len(lb.classes_))

# opt = PsoOptimizer()
opt = SGD(lr=hyperparameters.init_lr, decay=hyperparameters.init_lr / hyperparameters.epochs)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

""""And before I go, there is one great new coming up soon. Tensorflow 2.2 will allow you to feed your own train_on_batch and validate_on_batch functions to the original .fit API. """

# train the network
# H = model.fit(train_x, train_y, batch_size=hyperparameters.batch_size, validation_data=(test_x, test_y),
#             steps_per_epoch=len(train_x) // hyperparameters.batch_size, epochs=hyperparameters.epochs)

### The Custom Loop
#@tf.function
def train_on_batch(X, y):
    with GradientTape() as tape:
        ŷ = model(X, training=True)
        weights = model.get_weights()
        loss_value = loss(y, ŷ)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    weights2 = model.get_weights();
    model.set_weights(weights)


# The validate_on_batch function
# Find out how the model works
# @tf.function
def validate_on_batch(X, y):
    ŷ = model(X, training=False)
    loss_value = loss(y, ŷ)
    return loss_value


# Putting it all together
loss = categorical_crossentropy
optimizer = Adam(0.001)

train_data = Dataset.from_tensor_slices((train_x, train_y)).shuffle(buffer_size=len(train_x)).batch(
    hyperparameters.batch_size)
test_data = Dataset.from_tensor_slices((test_x, test_y)).shuffle(buffer_size=len(test_x)).batch(
    hyperparameters.batch_size)

# Enumerating the Dataset
best_loss = 99999
for epoch in range(0, hyperparameters.epochs):
    for batch, (X, y) in enumerate(train_data):
        train_on_batch(X, y)
        print('\rEpoch [%d/%d] Batch: %d%s' % (epoch + 1, hyperparameters.epochs, batch, '.' * (batch % 10)), end='')

    val_loss = np.mean([np.mean(validate_on_batch(X, y)) for (X, y) in test_data])
    print('. Validation Loss: ' + str(val_loss))
    if val_loss < best_loss:
        model.save_weights('model.h5')
        best_loss = val_loss

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
