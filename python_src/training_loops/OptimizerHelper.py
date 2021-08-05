import numpy as np
from tensorflow.python.keras.layers import Conv2D, Dense
from tensorflow.python.keras.metrics import TrueNegatives, TruePositives, FalsePositives, \
    FalseNegatives, BinaryCrossentropy, CategoricalCrossentropy

from metrics.MetricsUtil import iou_coef, dice_coef


def calc_solution_fitness(weights, model, loss_metric, X, y):
    set_trainable_weights(model, weights)
    ŷ = model(X, training=True)

    tn = TrueNegatives()
    tp = TruePositives()
    fp = FalsePositives()
    fn = FalseNegatives()

    loss = loss_metric(y, ŷ).numpy()
    tn_score = tn(y, ŷ).numpy()
    tp_score = tp(y, ŷ).numpy()
    fp_score = fp(y, ŷ).numpy()
    fn_score = fn(y, ŷ).numpy()

    precision = tp_score / (tp_score + fn_score)
    specificity = tn_score / (tn_score + fp_score)

    fpr = fp_score / (tn_score + fn_score + tp_score + fp_score)
    return (fpr) + (2 * loss) + (1 - specificity) + (1 - precision)


def calc_seg_fitness(weights, model, loss_metric, X, y):
    set_trainable_weights(model, weights, as_numpy_array=True)
    ŷ = model(X, training=True)
    return 2 - (iou_coef(y, ŷ)+dice_coef(y, ŷ))


def determine_loss_function_based_on_fitness_function(fitness_function):
    if fitness_function == calc_solution_fitness:
        return BinaryCrossentropy()
    else:
        return CategoricalCrossentropy()


def get_trainable_weights(model, keras_layers=(Dense, Conv2D), as_numpy_array=True):
    weights = []
    for layer in model.layers:
        if (layer.trainable != True or len(layer.trainable_weights) == 0 or layer.name == 'predictions'):
            pass
        if isinstance(layer, keras_layers):
            weights.append(layer.get_weights()) if as_numpy_array else weights.append(layer.trainable_weights)
    if as_numpy_array:
        return np.array(weights, dtype=object)
    else:
        return weights


def set_trainable_weights(model, weights, keras_layers=(Dense, Conv2D), as_numpy_array=True):
    i = 0
    for layer in model.layers:
        if (layer.trainable != True or len(layer.weights) == 0 or layer.name == 'predictions'):
            pass
        if isinstance(layer, keras_layers):
            if as_numpy_array:
                layer.set_weights(weights[i].tolist())
            else:
                layer._trainable_weights = weights[i]
            i += 1
    return model

