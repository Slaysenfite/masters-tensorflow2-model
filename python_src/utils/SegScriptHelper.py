import tensorflow as tf
from numpy import expand_dims


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(model, test_x, test_y, index=2, title='pred.png'):
    import matplotlib.pyplot as plt
    image = test_x[index]
    image = expand_dims(image, axis=0)
    pred_mask = model.predict(image)
    display(plt, [test_x[index], test_y[index], pred_mask[0]], title)


def display(plt, display_list, file_title):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.savefig(file_title)
    plt.clf()
    # plt.show()


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask
