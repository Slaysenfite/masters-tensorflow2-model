import time
from datetime import timedelta

from training_loops.PsoOptimizer import PsoEnv
from training_loops.TrainingHelper import print_metrics, append_epoch_metrics, reset_metrics, validate_on_batch, \
    prepare_metrics, batch_data_set, generate_tf_history


def train_on_batch(model, X, y, accuracy_metric, loss_metric):
    pso = PsoEnv(swarm_size=8, iterations=4, model=model, X=X, y=y)
    model = pso.get_pso_model()

    ŷ = model(X, training=True)
    # Calculate loss after pso weight updating
    accuracy = accuracy_metric(y, ŷ)
    loss = loss_metric(y, ŷ)

    # Update training metric.
    return accuracy.numpy(), loss.numpy()


### The Custom Loop For The PSO based optimizer
def training_loop(model, hyperparameters, train_x, train_y, test_x, test_y, callbacks):
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

            train_acc_score, train_loss_score = train_on_batch(model, X, y, train_acc_metric, train_loss_metric)

        # Run a validation loop at the end of each epoch.
        for (X, y) in test_data:
            val_acc_score, val_loss_score = validate_on_batch(model, X, y, val_acc_metric, val_loss_metric)
        print('\rValidation accuracy: %.4f' % (float(val_acc_score)) + 'Validation loss: %.4f' % (float(val_loss_score)))

        # Display metrics at the end of each epoch.
        print_metrics(train_acc_score, train_loss_score, val_acc_score, val_loss_score)

        # Append metrics
        append_epoch_metrics(accuracy, loss, train_acc_score, train_loss_score, val_acc_score, val_accuracy, val_loss,
                             val_loss_score)

        # Reset metrics
        reset_metrics(train_acc_metric, val_acc_metric)

        print('\rTime taken: ' + str(timedelta(seconds=(time.time() - start_time))))

    return generate_tf_history(accuracy, hyperparameters, loss, model, val_accuracy, val_loss)
