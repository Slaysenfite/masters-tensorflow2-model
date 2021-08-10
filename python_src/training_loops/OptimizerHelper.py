import numpy as np
import tensorflow as tf
import tensorflow.python.framework.ops as tf_ops
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
    set_trainable_weights(model, weights)
    ŷ = model(X, training=True)
    return 2 - (iou_coef(y, ŷ)+dice_coef(y, ŷ))


def determine_loss_function_based_on_fitness_function(fitness_function):
    if fitness_function == calc_solution_fitness:
        return BinaryCrossentropy()
    else:
        return CategoricalCrossentropy()


def get_trainable_weights(model, keras_layers=(Dense, Conv2D)):
    weights = []
    for layer in model.layers:
        if layer.trainable != True or len(layer.trainable_weights) == 0 or layer.name == 'predictions':
            pass

        if isinstance(layer, keras_layers):
            weights.append(layer.weights)
    return weights


def set_trainable_weights(model, weights, keras_layers=(Dense, Conv2D)):
    i = 0
    for layer in model.layers:
        if layer.trainable != True or len(layer.weights) == 0 or layer.name == 'predictions':
            pass
        if isinstance(layer, keras_layers):
            np_weights = np.zeros_like(layer.get_weights())
            for n in range(len(np_weights)):
                np_weights[n] = np.zeros_like(layer.get_weights()[n])
            for c in range(len(layer.weights)):
                if isinstance(weights[i][c], tf.Tensor) or isinstance(weights[i][c], tf_ops.Tensor):
                    np_weights[c] = weights[i][c].numpy()
                elif isinstance(weights[i][c], np.ndarray):
                    np_weights[c] = weights[i][c]
                else:
                    np_weights[c] = weights[i][c].value().numpy()
            layer.set_weights(np_weights)
            i += 1
    return model


def convert_tenor_weights_to_tf_variable(weights):
    for r in range(len(weights)):
        for c in range(len(weights[r])):
            if isinstance(weights[r][c], tf.Tensor) or isinstance(weights[r][c], tf_ops.Tensor):
                weights[r][c] = tf.Variable(weights[r][c])
    return weights


def perform_tensor_operations(operation_function, tensor_1, tensor_2):
    new_tensor = []
    for i in range(len(tensor_1)):
        vars = []
        for n in range(len(tensor_1[i])):
            vars.append(operation_function(tensor_1[i][n], tensor_2[i][n]))
        new_tensor.append(
            vars
        )
    return new_tensor


def add_three_tensors(tensor_1, tensor_2, tensor_3):
    new_tensor = []
    for i in range(len(tensor_1)):
        vars = []
        for n in range(len(tensor_1[i])):
            vars.append(tensor_1[i][n] + tensor_2[i][n] + tensor_3[i][n])
        new_tensor.append(
            vars
        )
    return new_tensor


def create_empty_tensor_with_same_shape(tensor):
    new_tensor = []
    for i in range(len(tensor)):
        vars = []
        for x in tensor:
            vars.append(0 * x)
        new_tensor.append(
            vars
        )
    return new_tensor
