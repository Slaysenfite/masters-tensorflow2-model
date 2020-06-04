from random import random, randint, seed
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, \
    Dense


seed(1)
rand_seed = randint(0, 10)
seed(rand_seed)


class Particle:
    def __init__(self, position, velocity):
        self.velocity = velocity
        self.position = position
        self.gbest = position
        self.pbest = position


#TODO: Implement this method
def calculate_loss(weights):
    pass


class Position:
    def __init__(self, weights):
        self.weights = weights
        self.loss = calculate_loss(weights)


def update_velocity(particle, inertia_weight, acc_c):
    # TODO: Look into clamping the velocity
    return (inertia_weight)(particle.velocity) \
           + (acc_c[0])(random())(particle.pbest - particle.position) \
           + (acc_c[1])(random())(particle.gbest - particle.position)


def update_position(particle, inertia_weight, acc_c):
    return particle.position + update_position(particle, inertia_weight, acc_c)

# def calculate_loss():


class VggOneBlock:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        # FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
