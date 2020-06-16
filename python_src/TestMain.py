import sys

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.data import Dataset
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.layers import Conv2D, Dense
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import CategoricalCrossentropy, CategoricalAccuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from configurations.GConstants import create_required_directories
from metrics.MetricsReporter import MetricReporter
from model.DataSet import mias_data_set as data_set
from model.Hyperparameters import hyperparameters
from optimizers.OptimizerHelper import Particle, calc_min_loss, C2, C1, INERTIA, update_velocity, \
    update_position, VggOneBlockFunctional
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

model = VggOneBlockFunctional.build(IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2], classes=len(lb.classes_))

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

from random import seed, uniform

seed(1)


### The Custom Loop
# @tf.function
def train_on_batch(X, y):
    ŷ = model(X, training=True)
    loss_value = loss_function(y, ŷ)
    losses = []

    weights = get_trainable_weights(model)
    losses.append(calc_position_loss(weights, model, train_loss_metric, X, y))

    swarm = initialize_swarm(15, weights, model, train_loss_metric, X, y)
    calculate_initial_losses(swarm, model, train_loss_metric, X, y)
    update_positions(swarm)

    # Calculate loss after pso weight updating
    train_acc_metric(y, ŷ)
    train_loss_metric(y, ŷ)

    # Update training metric.
    return train_acc_metric.result().numpy(), train_loss_metric.result().numpy()


class PsoEnv():
    def __init__(self, iterations, swarm_size, inertia, c_1, c_2, weights, model, loss_metric, X, y):
        self.iterations = iterations
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.c_1 = c_1
        self.c_2 = c_2
        self.weights = weights
        self.model = model
        self.loss_metric = loss_metric
        self.X = X
        self.y = y


def calc_position_loss(weights, model, loss_metric, X, y):
    set_trainable_weights(model, weights)
    ŷ = model(X, training=True)
    loss_metric(y, ŷ)
    return train_loss_metric.result().numpy()


def calculate_initial_losses(particles, model, loss_metric, X, y):
    best_initial_loss = calc_min_loss(particles)[2]
    update_gbest(best_initial_loss, particles)


def update_gbest(best_initial_loss, particle):
    for particle in particle:
        particle.gbest = best_initial_loss


def update_positions(particles):
    for particle in particles:
        particle.velocity = update_velocity(particle, INERTIA, [C1, C2])
        new_pos = update_position(particle, INERTIA, [C1, C2])
        if new_pos < particle.pbest:
            particle.pbest = new_pos


def initialize_swarm(swarm_size, weights, model, loss_metric, X, y):
    particles = [None] * swarm_size
    particles[0] = Particle(weights, calc_position_loss(weights, model, loss_metric, X, y))
    for p in range(1, swarm_size):
        new_weights = [None] * len(weights)
        for i in range(0, len(weights)):
            new_weights[i] = weights[i] * uniform(0, 1)
        initial_loss = calc_position_loss(new_weights, model, loss_metric, X, y)
        particles[p] = Particle(new_weights, initial_loss)
    return particles


def get_trainable_weights(model):
    weights = []
    for layer in model.layers:
        if (layer.trainable != True or len(layer.trainable_weights) == 0):
            pass
        if isinstance(layer, (Conv2D, Dense)):
            t_weights_for_layer = []
            t_weights_for_layer.append(layer.weights[0].numpy())
            weights.append(t_weights_for_layer)
    return weights


def set_trainable_weights(model, weights):
    i = 0
    for layer in model.layers:
        if (layer.trainable != True or len(layer.weights) == 0):
            pass
        if isinstance(layer, (Conv2D, Dense)):
            layer.weights[0].numpy = weights[i]
            i += 1
    return model


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
