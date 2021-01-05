import os

from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from configurations.Enums import LearningOptimization


def create_required_directories():
    os.makedirs(output_dir, 0o777, True)
    os.makedirs(output_dir + 'figures/', 0o777, True)
    os.makedirs(output_dir + 'model/', 0o777, True)


IMAGE_DIMS = (256, 256, 3)

output_dir = 'output/'

FIGURE_OUTPUT = output_dir + 'figures/'
MODEL_OUTPUT = output_dir + 'model/'


class Hyperparameters:
    def __init__(self, epochs, init_lr, batch_size,
                 learning_optimization, dropout):
        self.epochs = epochs
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_optimization = learning_optimization

    def report_hyperparameters(self):
        report = '*** Script Hyperparameters ***\n'
        report += ' Epochs: {}\n'.format(self.epochs)
        report += ' Initial learning rate: {}\n'.format(self.init_lr)
        report += ' Batch size: {}\n'.format(self.batch_size)
        report += ' Dropout: {}\n'.format(self.dropout)
        report += ' Learning optimization: {}\n'.format(self.learning_optimization)

        return report


def create_standard_hyperparameter_singleton():
    return Hyperparameters(
        50,
        5e-3,
        32,
        LearningOptimization.ADAM,
        0.25
    )

def create_mnist_hyperparameter_singleton():
    return Hyperparameters(
        10,
        5e-3,
        2000,
        LearningOptimization.SGD,
        0.25
    )

def create_pso_hyperparameter_singleton():
    return Hyperparameters(
        4,
        0,
        96,
        LearningOptimization.PSO,
        0.25
    )

def create_hybrid_hyperparameter_singleton():
    return Hyperparameters(
        1,
        5e-3,
        96,
        LearningOptimization.HYBRID,
        0.25
    )

def create_callbacks():
    return [
        EarlyStopping(
            monitor='val_loss', min_delta=0.0001, patience=15, verbose=1, mode='min',
            baseline=1.00, restore_best_weights=False),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='min',
            min_delta=0.0001, cooldown=0, min_lr=0)
    ]


hyperparameters = create_standard_hyperparameter_singleton()
mnist_hyperparameters = create_mnist_hyperparameter_singleton()
pso_hyperparameters = create_pso_hyperparameter_singleton()
hybrid_hyperparameters = create_hybrid_hyperparameter_singleton()
