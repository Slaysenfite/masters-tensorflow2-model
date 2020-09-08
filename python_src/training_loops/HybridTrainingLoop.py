import time

from tensorflow import GradientTape

from training_loops.PsoOptimizer import PsoEnv
from training_loops.TrainingHelper import print_metrics, append_epoch_metrics, reset_metrics, validate_on_batch, \
    batch_data_set, prepare_metrics, generate_tf_history


def train_on_batch(model, optimizer, X, y, previous_loss_scalar, accuracy_metric, loss_metric):
    loss, ŷ = compute_loss_via_pso(X, loss_metric, model, y)
    if loss > previous_loss_scalar:
        print('[TRAINING INFO] Compute gradients and calculate new loss')
        gd_loss, ŷ_gd = compute_loss_via_gradient_descent(X, loss_metric, model, optimizer, y)
        if gd_loss < loss:
            print('[TRAINING INFO] Using gradient descent weights')
            loss = gd_loss
            ŷ = ŷ_gd

    # Calculate loss after pso weight updating
    accuracy = accuracy_metric(y, ŷ)

    # Update training metric.
    return accuracy.numpy(), loss.numpy()


def compute_loss_via_pso(X, loss_metric, model, y):
    pso = PsoEnv(5, 8, model, X, y)
    model = pso.get_pso_model()
    ŷ = model(X, training=True)
    loss = loss_metric(y, ŷ)
    return loss, ŷ


def compute_loss_via_gradient_descent(X, loss_metric, model, optimizer, y):
    with GradientTape() as tape:
        ŷ = model(X, training=True)
        loss_value = loss_metric(y, ŷ)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    ŷ = model(X, training=True)
    loss = loss_metric(y, ŷ)
    return loss, ŷ


### The Custom Loop For The SGD-PSO based optimizer
def training_loop(model, optimizer, hyperparameters, train_x, train_y, test_x, test_y):
    # Separate into batches
    test_data, train_data = batch_data_set(hyperparameters, test_x, test_y, train_x, train_y)

    loss = []
    accuracy = []
    val_loss = []
    val_accuracy = []

    # Start timer
    start_time = time.time()

    # Enumerating the Dataset
    for epoch in range(0, hyperparameters.epochs):

        # Prepare the metrics.
        train_acc_metric, train_loss_metric, val_acc_metric, val_loss_metric = prepare_metrics()
        previous_loss_scalar = 999999999;

        for batch, (X, y) in enumerate(train_data):
            print('\rEpoch [%d/%d] Batch: %d%s \n' % (epoch + 1, hyperparameters.epochs, batch, '.' * (batch % 10)),
                  end='')

            train_acc_score, train_loss_score = train_on_batch(model, optimizer, X, y, previous_loss_scalar,
                                                               train_acc_metric,
                                                               train_loss_metric)
            previous_loss_scalar = train_loss_score

        # Run a validation loop at the end of each epoch.
        for (X, y) in test_data:
            val_acc_score, val_loss_score = validate_on_batch(model, X, y, val_acc_metric, val_loss_metric)
        print(
            "Validation accuracy: %.4f" % (float(val_acc_score)) + "\n Validation loss: %.4f" % (float(val_loss_score)))

        # Display metrics at the end of each epoch.
        print_metrics(train_acc_score, train_loss_score, val_acc_score, val_loss_score)

        # Append metrics
        append_epoch_metrics(accuracy, loss, train_acc_score, train_loss_score, val_acc_score, val_accuracy, val_loss,
                             val_loss_score)

        # Reset metrics
        reset_metrics(train_acc_metric, val_acc_metric)

    print("Time taken: %.2fs" % (time.time() - start_time))

    return generate_tf_history(accuracy, hyperparameters, loss, model, val_accuracy, val_loss)
