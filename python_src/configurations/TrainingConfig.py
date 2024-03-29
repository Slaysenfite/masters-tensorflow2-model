import os
from os.path import expanduser

from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

home = expanduser("~")
output_dir = 'output/'

FIGURE_OUTPUT = output_dir + 'figures/'
SEGMENTATION_OUTPUT = output_dir + 'segmentation/'
HEATMAPS_OUTPUT = output_dir + 'heatmaps/'
MODEL_OUTPUT = ROOT_DIRECTORY = home + '/data/models/'


def create_required_directories():
    os.makedirs(output_dir, 0o777, True)
    os.makedirs(output_dir + 'figures/', 0o777, True)
    os.makedirs(output_dir + 'segmentation/', 0o777, True)
    os.makedirs(output_dir + 'heatmaps/', 0o777, True)
    os.makedirs(MODEL_OUTPUT, 0o777, True)


IMAGE_DIMS = (128, 128, 3)

home = expanduser("~")
output_dir = 'output/'

FIGURE_OUTPUT = output_dir + 'figures/'
MODEL_OUTPUT = ROOT_DIRECTORY = home + '/data/models/'
SEGMENTATION_OUTPUT = output_dir + 'segmentation/'
HEATMAP_OUTPUT = output_dir + 'heatmaps/'


class Hyperparameters:
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.lr = 0.001
        self.batch_size = batch_size
        self.dropout_prob = 0.25
        self.learning_optimization = 'sgd'
        self.meta_heuristic = 'none'
        self.experiment_id = 'na'
        self.augmentation = False
        self.preloaded_weights = False
        self.kernel_initializer = 'he_uniform'
        self.weights_of_experiment_id = None
        self.tf_fit = True
        self.l2 = 0.00001
        self.num_layers_for_optimization = 10
        self.dataset_segment = "All Segments"

    def report_hyperparameters(self):
        report = '*** Script Hyperparameters ***\n'
        report += ' Experiment Id: {}\n'.format(self.experiment_id)
        report += ' Epochs: {}\n'.format(self.epochs)
        report += ' Initial learning rate: {}\n'.format(self.lr)
        report += ' Batch size: {}\n'.format(self.batch_size)
        report += ' Dropout: {}\n'.format(self.dropout_prob)
        report += ' Learning optimization: {}\n'.format(self.learning_optimization)
        report += ' Meta-heuristic used: {}\n'.format(self.meta_heuristic)
        report += ' Meta-heuristic layer optimized: {}\n'.format(self.num_layers_for_optimization)
        report += ' Data augmentation: {}\n'.format(self.augmentation)
        report += ' Preloaded weights: {}\n'.format(self.preloaded_weights)
        report += ' Kernel initializer: {}\n'.format(self.kernel_initializer)
        report += ' Existing Weights Exp Id: {}\n'.format(self.weights_of_experiment_id)
        report += ' TF Fit Training: {}\n'.format(self.tf_fit)
        report += ' L2: {}\n'.format(self.l2)
        report += ' Dataset subset: {}\n'.format(self.dataset_segment)

        return report


def create_standard_hyperparameter_singleton():
    return Hyperparameters(
        50,
        32
    )


def create_mnist_hyperparameter_singleton():
    return Hyperparameters(
        10,
        32
    )


def create_callbacks(hyperparameters):
    return [
        EarlyStopping(
            monitor='val_loss', min_delta=0.0001, patience=20, verbose=1, mode='min',
            baseline=1.00, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='min',
            min_delta=0.001, cooldown=0, min_lr=0.00001),
        ModelCheckpoint(
            '{}{}.h5'.format(MODEL_OUTPUT, hyperparameters.experiment_id), monitor='val_loss', verbose=0,
            save_best_only=True, save_weights_only=True, mode='min', save_freq='epoch',
            options=None
        )
    ]


hyperparameters = create_standard_hyperparameter_singleton()
mnist_hyperparameters = create_mnist_hyperparameter_singleton()
