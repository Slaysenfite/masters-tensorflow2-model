import time
from datetime import timedelta

from tensorflow import GradientTape
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.metrics import Precision, Recall

from training_loops.OptimizerHelper import get_trainable_weights, set_trainable_weights
from training_loops.PsoOptimizer import PsoEnv
from training_loops.TrainingHelper import print_metrics, append_epoch_metrics, reset_metrics, \
    batch_data_set, prepare_metrics, generate_tf_history


def train_on_batch(model, optimizer, X, y, accuracy_metric, loss_metric, pso_layer, gd_layer):
    if ((pso_layer == (Dense)) & (gd_layer == (Conv2D))) | ((pso_layer == (Conv2D, Dense)) & (gd_layer == (Conv2D, Dense))):
        apply_gradient_descent(X, gd_layer, loss_metric, model, optimizer, y)
        model = apply_swarm_optimization(X, model, pso_layer, y)
    elif (pso_layer == (Conv2D, Dense)) & (gd_layer == None):
        model = apply_swarm_optimization(X, model, pso_layer, y)
    else:
        model = apply_swarm_optimization(X, model, pso_layer, y)
        apply_gradient_descent(X, gd_layer, loss_metric, model, optimizer, y)
    ŷ = model(X, training=True)
    # Calculate loss after pso weight updating
    precision_metric = Precision()
    recall_metric = Recall()
    accuracy = accuracy_metric(y, ŷ)
    loss = loss_metric(y, ŷ)
    precision = precision_metric(y, ŷ)
    recall = recall_metric(y, ŷ)
    # Update training metric.
    return accuracy.numpy(), loss.numpy(), precision.numpy(), recall.numpy()


def apply_swarm_optimization(X, model, pso_layer, y):
    pso = PsoEnv(swarm_size=10, iterations=5, model=model, X=X, y=y, layers_to_optimize=pso_layer)
    model = pso.get_pso_model()
    return model


def apply_gradient_descent(X, gd_layer, loss_metric, model, optimizer, y):
    with GradientTape() as tape:
        ŷ = model(X, training=True)
        loss_value = loss_metric(y, ŷ)
    gd_weights = get_trainable_weights(model, keras_layers=gd_layer, as_numpy_array=False)
    grads = tape.gradient(loss_value, gd_weights[0])
    optimizer.apply_gradients(zip(grads, gd_weights[0]))
    set_trainable_weights(model, gd_weights, gd_layer, as_numpy_array=False)

# The validate_on_batch function
# Find out how the model works
# @tf.function
def validate_on_batch(model, X, y, accuracy_metric, loss_metric):
    ŷ = model(X, training=False)
    precision_metric = Precision()
    recall_metric = Recall()
    accuracy = accuracy_metric(y, ŷ)
    loss = loss_metric(y, ŷ)
    precision = precision_metric(y, ŷ)
    recall = recall_metric(y, ŷ)
    return accuracy.numpy(), loss.numpy(), precision.numpy(), recall.numpy()


# The Custom Loop For The SGD-PSO based optimizer
def training_loop(model, optimizer, hyperparameters, train_x, train_y, test_x, test_y, pso_layer=(Dense), gd_layer=(Conv2D)):
    # Separate into batches
    test_data, train_data = batch_data_set(hyperparameters, test_x, test_y, train_x, train_y)

    loss = []
    accuracy = []
    val_loss = []
    val_accuracy = []

    # Enumerating the Dataset
    for epoch in range(0, hyperparameters.epochs):
        start_time = time.time()

        # Prepare the metrics.
        train_acc_metric, train_loss_metric, val_acc_metric, val_loss_metric = prepare_metrics()

        for batch, (X, y) in enumerate(train_data):
            print('\rEpoch [%d/%d] Batch: %d%s \n' % (epoch + 1, hyperparameters.epochs, batch, '.' * (batch % 10)),
                  end='')

            train_acc_score, train_loss_score, train_precision_score, train_recall_score = train_on_batch(model,
                                                                                                          optimizer,
                                                                                                          X, y,
                                                                                                          train_acc_metric,
                                                                                                          train_loss_metric,
                                                                                                          pso_layer,
                                                                                                          gd_layer)

        # Run a validation loop at the end of each epoch.
        for (X, y) in test_data:
            val_acc_score, val_loss_score, val_precision_score, val_recall_score = validate_on_batch(model, X, y,
                                                                                                     val_acc_metric,
                                                                                                     val_loss_metric)

        # Display metrics at the end of each epoch.
        print_metrics(train_acc_score, train_loss_score, train_precision_score, train_recall_score, val_acc_score,
                      val_loss_score, val_precision_score, val_recall_score)

        # Append metrics
        append_epoch_metrics(accuracy, loss, train_acc_score, train_loss_score, val_acc_score, val_accuracy, val_loss,
                             val_loss_score)

        # Reset metrics
        reset_metrics(train_acc_metric, val_acc_metric)

        print('\rTime taken: ' + str(timedelta(seconds=(time.time() - start_time))))

    return generate_tf_history(accuracy, hyperparameters, loss, model, val_accuracy, val_loss)
