from array import array

from numpy.ma import array
from scipy.spatial.distance import dice
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import MeanIoU


def iou_coef(y_true, y_pred, num_classes=2, smooth=1):
    m = MeanIoU(num_classes)
    m.update_state(y_true, y_pred)
    return m.result().numpy()


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = array(K.flatten(y_true))
    y_pred_f = array(K.flatten(y_pred))
    return dice(y_true_f, y_pred_f)
