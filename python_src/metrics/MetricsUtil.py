from array import array

import tensorflow as tf
from scipy.spatial.distance import dice
from tensorflow.python.keras import backend as K
from numpy.ma import array


def iou_coef(y_true, y_pred, smooth=1):
    m = tf.keras.metrics.MeanIoU(num_classes=3)
    m.update_state(y_true, y_pred)
    return m.result().numpy()


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = array(K.flatten(y_true))
    y_pred_f = array(K.flatten(y_pred))
    return dice(y_true_f, y_pred_f)
