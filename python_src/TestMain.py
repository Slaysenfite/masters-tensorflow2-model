import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import GradientTape
from tensorflow.python.data import Dataset
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import CategoricalCrossentropy, Accuracy, CategoricalAccuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from configurations.GConstants import create_required_directories
from metrics.MetricsReporter import MetricReporter
from model.DataSet import mias_data_set as data_set
from model.Hyperparameters import hyperparameters
from optimizers.OptimizerHelper import VggOneBlock
from utils.Emailer import results_dispatch
from utils.ImageLoader import load_rgb_images
from utils.ScriptHelper import read_cmd_line_args, generate_script_report

IMAGE_DIMS = (64, 64, 3)

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
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.3, train_size=0.7,
                                                      random_state=42)

# binarize the class labels
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

model = VggOneBlock.build(IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2], classes=len(lb.classes_))

# opt = PsoOptimizer()
opt = SGD(lr=hyperparameters.init_lr, decay=hyperparameters.init_lr / hyperparameters.epochs)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Separate into batches
train_data = Dataset.from_tensor_slices((train_x, train_y)).shuffle(buffer_size=len(train_x)).batch(
    hyperparameters.batch_size)
test_data = Dataset.from_tensor_slices((test_x, test_y)).shuffle(buffer_size=len(test_x)).batch(
    hyperparameters.batch_size)

# Prepare the metrics.
train_acc_metric = CategoricalAccuracy()
val_acc_metric = CategoricalAccuracy()
train_loss_metric = CategoricalCrossentropy()
val_loss_metric = CategoricalCrossentropy()
loss_function = categorical_crossentropy
optimizer = Adam(0.000)
H = History()
H.set_model(model)
H.set_params({
    'batch_size': hyperparameters.batch_size,
    'epochs': hyperparameters.epochs,
    'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy']
})
history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
loss = []
accuracy = []
val_loss = []
val_accuracy = []


### The Custom Loop
# @tf.function
def train_on_batch(X, y):
    with GradientTape() as tape:
        ŷ = model(X, training=True)
        weights = model.get_weights()
        loss_value = loss_function(y, ŷ)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    #model.set_weights(weights)
    train_acc_metric(y, ŷ)
    train_loss_metric(y, ŷ)

    # Update training metric.
    return train_acc_metric.result().numpy(), train_loss_metric.result().numpy()


# The validate_on_batch function
# Find out how the model works
# @tf.function
def validate_on_batch(X, y):
    ŷ = model(X, training=False)
    val_acc_metric(y, ŷ)
    val_loss_metric(y, ŷ)

    return val_acc_metric.result().numpy(), val_loss_metric.result().numpy()

# Enumerating the Dataset
best_loss = 99999
for epoch in range(0, hyperparameters.epochs):
    for batch, (X, y) in enumerate(train_data):
        train_acc_score, train_loss_score = train_on_batch(X, y)
        val_acc_score, val_loss_score = validate_on_batch(X, y)
        print('\rEpoch [%d/%d] Batch: %d%s' % (epoch + 1, hyperparameters.epochs, batch, '.' * (batch % 10)), end='')

    # Display metrics at the end of each epoch.
    print('Training acc over epoch: %s' % (float(train_acc_score)))
    print('Training loss over epoch: %s' % (float(train_loss_score)))
    print('Validation acc over epoch: %s' % (float(val_acc_score)))
    print('Validation loss over epoch: %s' % (float(val_loss_score)))

    # Reset metrics
    train_acc_metric.reset_states()
    train_loss_metric.reset_states()
    val_acc_metric.reset_states()
    val_loss_metric.reset_states()

    loss.append(train_loss_score)
    accuracy.append(train_acc_score)
    val_loss.append(val_loss_score)
    val_accuracy.append(val_acc_score)

    # val_loss_float = np.mean([np.mean(validate_on_batch(X, y)) for (X, y) in test_data])
    # print('. Validation Loss: ' + str(val_loss_float))
    # if val_loss_float < best_loss:
    #     # model.save_weights('model.h5')
    #     best_loss = val_loss

# Update history object
history['loss'] = loss
history['accuracy'] = accuracy
history['val_loss'] = val_loss
history['val_accuracy'] = val_accuracy
H.history = history

# evaluate the network
print('[INFO] evaluating network...')

predictions = model.predict(test_x, batch_size=32)

print('[INFO] generating metrics...')

generate_script_report(H, test_y, predictions, data_set, hyperparameters, 'testnet')

reporter = MetricReporter(data_set.name, 'testnet')
cm1 = confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1))
reporter.plot_confusion_matrix(cm1, classes=data_set.class_names,
                               title='Confusion matrix, without normalization')

reporter.plot_roc(data_set.class_names, test_y, predictions)

reporter.plot_network_metrics(hyperparameters.epochs, H, "testnet")

print('[INFO] serializing network and label binarizer...')

reporter.save_model_to_file(model, lb)

print('[INFO] emailing result...')

results_dispatch(data_set.name, "testnet ")

print('[END] Finishing script...\n')
