import sys

import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input, Model, backend
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, \
    SeparableConv2D, MaxPooling2D, add, Conv2DTranspose

from configurations.DataSet import cbis_seg_data_set as data_set
from configurations.TrainingConfig import create_required_directories
from configurations.TrainingConfig import hyperparameters, IMAGE_DIMS
from utils.ImageLoader import load_seg_images
from utils.ScriptHelper import read_cmd_line_args

print('Python version: {}'.format(sys.version))
print('Tensorflow version: {}\n'.format(tf.__version__))
print('[BEGIN] Start script...\n')
hyperparameters, opt, data_set = read_cmd_line_args(hyperparameters, data_set)
print(' Image dimensions: {}\n'.format(IMAGE_DIMS))
print(hyperparameters.report_hyperparameters())

print('[INFO] Creating required directories...')
create_required_directories()

print('[INFO] Loading images...')
roi_data, roi_labels = load_seg_images(data_set, path_suffix='roi', image_dimensions=([IMAGE_DIMS[0], IMAGE_DIMS[1], 1]))
data, labels = load_seg_images(data_set, image_dimensions=([IMAGE_DIMS[0], IMAGE_DIMS[1], 1]))

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax


def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask


@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


# partition the data into training and testing splits using 70% of
# the data for training and the remaining 30% for testing
(train_x, test_x, train_y, test_y) = train_test_split(data, roi_data, test_size=0.3, train_size=0.7, random_state=42)

# def side_by_side():
#     for i in range(10):
#         f, axarr = plt.subplots(1, 1)
#         axarr[0, 0].imshow(train_x[i])
#         axarr[0, 1].imshow(train_y[i])
#         plt.show()
#

TRAIN_LENGTH = len(train_x)
BATCH_SIZE = hyperparameters.batch_size
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE


def get_model(img_size, num_classes):
    inputs = Input(shape=img_size)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same")(residual)
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
backend.clear_session()

# Build model
model = get_model([IMAGE_DIMS[0], IMAGE_DIMS[1], 1], len(data_set.class_names))
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]


def show_predictions(train_x=train_x, train_y=train_y, index=1):
    image = train_x[index]
    mask = train_y[index]
    pred_mask = model.predict(image)
    display([image, mask, create_mask(pred_mask)])


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


model_history = model.fit(x=train_x, y=train_y,
                          epochs=hyperparameters.epochs,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=(test_x, test_y),
                          callbacks=[DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = hyperparameters.epochs

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

show_predictions(test_x, test_y, 3)
