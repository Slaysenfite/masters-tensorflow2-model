from tensorflow.python.data import Dataset
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import CategoricalAccuracy


def append_epoch_metrics(accuracy, loss, train_acc_score, train_loss_score, val_acc_score, val_accuracy, val_loss,
                             val_loss_score, val_precision, val_precision_score, val_recall, val_recall_score):
    loss.append(train_loss_score)
    accuracy.append(train_acc_score)
    val_loss.append(val_loss_score)
    val_accuracy.append(val_acc_score)
    val_precision.append(val_precision_score)
    val_recall.append(val_recall_score)


def reset_metrics(train_acc_metric, val_acc_metric):
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()


def print_metrics(train_acc_score, train_loss_score, train_precision_score, train_recall_score, val_acc_score,
                  val_loss_score, val_precision_score, val_recall_score):
    print('--- Training Scores ---')
    print('Training acc over epoch: %s' % (float(train_acc_score)))
    print('Training loss over epoch: %s' % (float(train_loss_score)))
    print('Training precision over epoch: %s' % (float(train_precision_score)))
    print('Training recall over epoch: %s' % (float(train_recall_score)))
    print('--- Validation Scores ---')
    print('Validation acc over epoch: %s' % (float(val_acc_score)))
    print('Validation loss over epoch: %s' % (float(val_loss_score)))
    print('Validation precision over epoch: %s' % (float(val_precision_score)))
    print('Validation recall over epoch: %s' % (float(val_recall_score)))


def generate_tf_history(model, hyperparameters, accuracy, loss, val_accuracy, val_loss, val_precision, val_recall):
    H = History()
    H.set_model(model)
    H.set_params({
        'batch_size': hyperparameters.batch_size,
        'epochs': hyperparameters.epochs,
        'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy', 'val_precision', 'val_recall']
    })
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [],'val_precision': [], 'val_recall': []}
    history['loss'] = loss
    history['accuracy'] = accuracy
    history['val_loss'] = val_loss
    history['val_accuracy'] = val_accuracy
    history['val_precision'] = val_precision
    history['val_recall'] = val_recall
    H.history = history
    return H


def prepare_metrics():
    train_acc_metric = CategoricalAccuracy()
    val_acc_metric = CategoricalAccuracy()
    train_loss_metric = CategoricalCrossentropy()
    val_loss_metric = CategoricalCrossentropy()
    return train_acc_metric, train_loss_metric, val_acc_metric, val_loss_metric


def batch_data_set(hyperparameters, test_x, test_y, train_x, train_y):
    train_data = Dataset.from_tensor_slices((train_x, train_y)).shuffle(buffer_size=len(train_x)).batch(
        hyperparameters.batch_size)
    test_data = Dataset.from_tensor_slices((test_x, test_y)).shuffle(buffer_size=len(test_x)).batch(
        hyperparameters.batch_size)
    return test_data, train_data

