from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import image_data_format
from tensorflow.python.keras.layers import BatchNormalization, Activation, Dropout, Conv2D, SeparableConv2D, \
    MaxPooling2D, add, GlobalAveragePooling2D, Dense, GlobalMaxPooling2D


def Xception(input_shape=None,
             pooling=None,
             weights=None,
             classes=3):
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=299,
                                      min_size=71,
                                      data_format=image_data_format(),
                                      require_flatten=True,
                                      weights=weights)

    img_input = Input(shape=input_shape)

    channel_axis = 1 if image_data_format() == 'channels_first' else -1

    x = Conv2D(32, (3, 3),
               strides=(2, 2),
               use_bias=False,
               name='block1_conv1')(img_input)
    x = BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)
    x = Dropout(0.25)(x)

    residual = Conv2D(128, (1, 1),
                      strides=(2, 2),
                      padding='same',
                      use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = SeparableConv2D(128, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block2_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = Dropout(0.25)(x)
    x = SeparableConv2D(128, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block2_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3),
                     strides=(2, 2),
                     padding='same',
                     name='block2_pool')(x)
    x = add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Dropout(0.25)(x)
    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block3_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = Dropout(0.25)(x)
    x = SeparableConv2D(256, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block3_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2),
                     padding='same',
                     name='block3_pool')(x)
    x = add([x, residual])

    residual = Conv2D(728, (1, 1),
                      strides=(2, 2),
                      padding='same',
                      use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Dropout(0.25)(x)
    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block4_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = Dropout(0.25)(x)
    x = SeparableConv2D(728, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block4_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2),
                     padding='same',
                     name='block4_pool')(x)
    x = add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = Dropout(0.25)(x)
        x = SeparableConv2D(728, (3, 3),
                            padding='same',
                            use_bias=False,
                            name=prefix + '_sepconv1')(x)
        x = BatchNormalization(axis=channel_axis,
                               name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = Dropout(0.25)(x)
        x = SeparableConv2D(728, (3, 3),
                            padding='same',
                            use_bias=False,
                            name=prefix + '_sepconv2')(x)
        x = BatchNormalization(axis=channel_axis,
                               name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = Dropout(0.25)(x)
        x = SeparableConv2D(728, (3, 3),
                            padding='same',
                            use_bias=False,
                            name=prefix + '_sepconv3')(x)
        x = BatchNormalization(axis=channel_axis,
                               name=prefix + '_sepconv3_bn')(x)

        x = add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = Dropout(0.25)(x)
    x = SeparableConv2D(728, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block13_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = Dropout(0.25)(x)
    x = SeparableConv2D(1024, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block13_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3),
                     strides=(2, 2),
                     padding='same',
                     name='block13_pool')(x)
    x = add([x, residual])

    x = SeparableConv2D(1536, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block14_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = Dropout(0.25)(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block14_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)
    x = Dropout(0.25)(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='xception')

    # Load weights.

    if weights is not None:
        model.load_weights(weights)

    return model
