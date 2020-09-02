import time

from tensorflow.python.eager.backprop import GradientTape

from training_loops.PsoOptimizer import PsoEnv
from training_loops.TrainingHelper import print_metrics, append_epoch_metrics, reset_metrics, validate_on_batch, \
    batch_data_set, prepare_metrics, generate_tf_history


def train_on_batch(model, optimizer, X, y, previous_loss_scalar, accuracy_metric, loss_metric):
    with GradientTape() as tape:
        logits = model(X, training=True)

        loss_value = loss_metric(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    ŷ = model(X, training=True)
    loss = loss_metric(y, ŷ)

    if loss.result().numpy() > previous_loss_scalar:
        pso = PsoEnv(5, 8, model, X, y)
        model = pso.get_pso_model()
        ŷ = model(X, training=True)
        new_loss = loss_metric(y, ŷ)
        if new_loss.result().numpy() < loss.result().numpy():
            loss = new_loss

    # Calculate loss after pso weight updating
    accuracy = accuracy_metric(y, ŷ)

    # Update training metric.
    return accuracy.result().numpy(), loss.result().numpy()


### The Custom Loop For The SGD-PSO based optimizer
# @tf.function
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
        print("Validation accuracy: %.4f" % (float(val_acc_score)) + "\n Validation loss: %.4f" % (float(val_loss_score)))

        # Display metrics at the end of each epoch.
        print_metrics(train_acc_score, train_loss_score, val_acc_score, val_loss_score)

        # Append metrics
        append_epoch_metrics(accuracy, loss, train_acc_score, train_loss_score, val_acc_score, val_accuracy, val_loss,
                             val_loss_score)

        # Reset metrics
        reset_metrics(train_acc_metric, train_loss_metric, val_acc_metric, val_loss_metric)

    print("Time taken: %.2fs" % (time.time() - start_time))

    return generate_tf_history(accuracy, hyperparameters, loss, model, val_accuracy, val_loss)
