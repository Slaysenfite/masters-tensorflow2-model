from tensorflow.python.keras import backend as K, Input, Model
from tensorflow.python.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, \
    Dense, Dropout
from tensorflow.python.keras.models import Sequential


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


class VggOneBlockFunctional:
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # CONV => RELU => POOL layer set
        inputs = Input(shape=inputShape)
        conv1 = Conv2D(64, (3, 3), padding="same")(inputs)
        act1 = Activation("relu")(conv1)
        bn1 = BatchNormalization(axis=chanDim)(act1)
        conv2 = Conv2D(64, (3, 3), padding="same")(bn1)
        act2 = Activation("relu")(conv2)
        bn2 = BatchNormalization(axis=chanDim)(act2)
        mp1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bn2)
        drop1 = Dropout(0.25)(mp1)

        # FC => RELU layers
        flat1 = Flatten()(drop1)
        dense1 = Dense(512)(flat1)
        act3 = Activation("relu")(dense1)
        bn3 = BatchNormalization()(act3)
        drop2 = Dropout(0.25)(bn3)

        # softmax classifier
        dense2 = Dense(classes)(drop2)
        final_act = Activation("softmax")(dense2)

        # return the constructed network architecture
        return Model(inputs=inputs, outputs=final_act)
