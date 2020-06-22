from tensorflow.python.data import Dataset
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy

from optimizers.PsoOptimizer import PsoEnv


def train_on_batch(model, X, y, accuracy_metric, loss_metric):
    pso = PsoEnv(5, 8, model, X, y)
    model = pso.get_pso_model()

    ŷ = model(X, training=True)
    # Calculate loss after pso weight updating
    accuracy_metric(y, ŷ)
    loss_metric(y, ŷ)

    # Update training metric.
    return accuracy_metric.result().numpy(), loss_metric.result().numpy()


# The validate_on_batch function
# Find out how the model works
# @tf.function
def validate_on_batch(model, X, y, accuracy_metric, loss_metric):
    ŷ = model(X, training=False)
    accuracy_metric(y, ŷ)
    loss_metric(y, ŷ)

    return accuracy_metric.result().numpy(), loss_metric.result().numpy()


### The Custom Loop
# @tf.function
def training_loop(model, hyperparameters, train_x, train_y, test_x, test_y):
    # Separate into batches
    train_data = Dataset.from_tensor_slices((train_x, train_y)).shuffle(buffer_size=len(train_x)).batch(
        hyperparameters.batch_size)
    test_data = Dataset.from_tensor_slices((test_x, test_y)).shuffle(buffer_size=len(test_x)).batch(
        hyperparameters.batch_size)

    H = History()
    H.set_model(model)
    H.set_params({
        'batch_size': hyperparameters.batch_size,
        'epochs': hyperparameters.epochs,
        'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy']
    })
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    loss = []
    accuracy = []
    val_loss = []
    val_accuracy = []

    # Enumerating the Dataset
    best_loss = 99999
    for epoch in range(0, hyperparameters.epochs):

        # Prepare the metrics.
        train_acc_metric = CategoricalAccuracy()
        val_acc_metric = CategoricalAccuracy()
        train_loss_metric = CategoricalCrossentropy(from_logits=True)
        val_loss_metric = CategoricalCrossentropy(from_logits=True)

        for batch, (X, y) in enumerate(train_data):
            print('\rEpoch [%d/%d] Batch: %d%s \n' % (epoch + 1, hyperparameters.epochs, batch, '.' * (batch % 10)),
                  end='')

            train_acc_score, train_loss_score = train_on_batch(model, X, y, train_acc_metric, train_loss_metric)
            val_acc_score, val_loss_score = validate_on_batch(model, X, y, val_acc_metric, val_loss_metric)

        # Display metrics at the end of each epoch.
        print_metrics(train_acc_score, train_loss_score, val_acc_score, val_loss_score)

        # Append metrics
        append_epoch_metrics(accuracy, loss, train_acc_score, train_loss_score, val_acc_score, val_accuracy, val_loss,
                             val_loss_score)

        # Reset metrics
        reset_metrics(train_acc_metric, train_loss_metric, val_acc_metric, val_loss_metric)

    # Update history object
    history['loss'] = loss
    history['accuracy'] = accuracy
    history['val_loss'] = val_loss
    history['val_accuracy'] = val_accuracy
    H.history = history

    return H


def append_epoch_metrics(accuracy, loss, train_acc_score, train_loss_score, val_acc_score, val_accuracy, val_loss,
                         val_loss_score):
    loss.append(train_loss_score)
    accuracy.append(train_acc_score)
    val_loss.append(val_loss_score)
    val_accuracy.append(val_acc_score)


def reset_metrics(train_acc_metric, train_loss_metric, val_acc_metric, val_loss_metric):
    train_acc_metric.reset_states()
    train_loss_metric.reset_states()
    val_acc_metric.reset_states()
    val_loss_metric.reset_states()


def print_metrics(train_acc_score, train_loss_score, val_acc_score, val_loss_score):
    print('Training acc over epoch: %s' % (float(train_acc_score)))
    print('Training loss over epoch: %s' % (float(train_loss_score)))
    print('Validation acc over epoch: %s' % (float(val_acc_score)))
    print('Validation loss over epoch: %s' % (float(val_loss_score)))
