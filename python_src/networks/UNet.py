
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow_examples.models.pix2pix import pix2pix

from configurations.TrainingConfig import IMAGE_DIMS

def build_unet(input_shape, num_classes):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=None,)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(input_shape)

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    pooling = tf.keras.layers.AveragePooling2D()(x)

    flatten = tf.keras.layers.Flatten()(pooling)
    fc = (tf.keras.layers.Dense(512))(flatten)
    act = tf.keras.layers.Activation('relu')(fc)

    dense = tf.keras.layers.Dense(num_classes)(act)
    final_layer = tf.keras.layers.Activation("softmax")(dense)

    return Model(inputs=inputs, outputs=final_layer)