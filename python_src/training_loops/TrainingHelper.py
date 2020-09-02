from tensorflow.python.data import Dataset
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy


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


def generate_tf_history(accuracy, hyperparameters, loss, model, val_accuracy, val_loss):
    H = History()
    H.set_model(model)
    H.set_params({
        'batch_size': hyperparameters.batch_size,
        'epochs': hyperparameters.epochs,
        'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy']
    })
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    history['loss'] = loss
    history['accuracy'] = accuracy
    history['val_loss'] = val_loss
    history['val_accuracy'] = val_accuracy
    H.history = history
    return H


def prepare_metrics():
    train_acc_metric = CategoricalAccuracy()
    val_acc_metric = CategoricalAccuracy()
    train_loss_metric = CategoricalCrossentropy(from_logits=True)
    val_loss_metric = CategoricalCrossentropy(from_logits=True)
    return train_acc_metric, train_loss_metric, val_acc_metric, val_loss_metric


def batch_data_set(hyperparameters, test_x, test_y, train_x, train_y):
    train_data = Dataset.from_tensor_slices((train_x, train_y)).shuffle(buffer_size=len(train_x)).batch(
        hyperparameters.batch_size)
    test_data = Dataset.from_tensor_slices((test_x, test_y)).shuffle(buffer_size=len(test_x)).batch(
        hyperparameters.batch_size)
    return test_data, train_data


# The validate_on_batch function
# Find out how the model works
# @tf.function
def validate_on_batch(model, X, y, accuracy_metric, loss_metric):
    ŷ = model(X, training=False)
    accuracy_metric(y, ŷ)
    loss_metric(y, ŷ)

    return accuracy_metric.result().numpy(), loss_metric.result().numpy()
