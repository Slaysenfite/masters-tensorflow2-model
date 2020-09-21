"""
Based on:

https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py

"""

from tensorflow.python.keras import backend as K, Input
from tensorflow.python.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, \
    Dense, ZeroPadding2D, concatenate, Cropping2D, UpSampling2D
from tensorflow.python.keras.models import Model

from model.Hyperparameters import hyperparameters


class UNet:

    @staticmethod
    def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2])
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw / 2), int(cw / 2) + 1
        else:
            cw1, cw2 = int(cw / 2), int(cw / 2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1])
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch / 2), int(ch / 2) + 1
        else:
            ch1, ch2 = int(ch / 2), int(ch / 2)

        return (ch1, ch2), (cw1, cw2)

    @staticmethod
    def build(img_shape, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        input_shape = img_shape
        chan_dim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            input_shape = (img_shape[2], img_shape[0], img_shape[1])
            chan_dim = 1

        concat_axis = 3
        inputs = Input(shape=input_shape)

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        ch, cw = UNet.get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch, cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        ch, cw = UNet.get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        ch, cw = UNet.get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        ch, cw = UNet.get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        ch, cw = UNet.get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        conv10 = Conv2D(classes, (1, 1))(conv9)

        flatten = Flatten()(conv10)
        fc = (Dense(512))(flatten)
        act = Activation('relu')(fc)
        batch_norm2 = BatchNormalization(axis=chan_dim)(act)

        dense = Dense(classes)(batch_norm2)
        final_layer = Activation("softmax")(dense)

        model = Model(inputs=inputs, outputs=final_layer)

        return model
