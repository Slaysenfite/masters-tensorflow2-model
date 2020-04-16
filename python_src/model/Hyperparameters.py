from model.Enums import ActivationFunctions, LearningOptimization, Pooling


class Hyperparameters:
    def __init__(self, epochs, init_lr, batch_size,
                 learning_optimization, dropout, activation_type,
                 pooling, num_of_hidden_layers, weight_initialization):
        self.epochs = epochs
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_optimization = learning_optimization
        self.activation_type = activation_type
        self.pooling = pooling
        self.num_of_hidden_layers = num_of_hidden_layers
        self.weight_initialization = weight_initialization

    def report_hyperparameters(self):
        report = '*** Script Hyperparameters ***\n'
        report += ' Epochs: {}\n'.format(self.epochs)
        report += ' Initial learning rate: {}\n'.format(self.init_lr)
        report += ' Batch size: {}\n'.format(self.batch_size)
        report += ' Dropout: {}\n'.format(self.dropout)
        report += ' Learning optimization: {}\n'.format(self.learning_optimization)
        report += ' Activation type: {}\n'.format(self.activation_type)
        report += ' Pooling: {}\n'.format(self.pooling)
        report += ' Number of hidden layers: {}\n'.format(self.num_of_hidden_layers)
        report += ' Pre-initialized weights: {}\n\n'.format(self.weight_initialization)

        return report


def create_hyperparameter_singleton():
    return Hyperparameters(
        3,
        5e-3,
        32,
        0.25,
        LearningOptimization.SGD,
        ActivationFunctions.RELU,
        Pooling.TWO_BY_TWO,
        4,
        False
    )


hyperparameters = create_hyperparameter_singleton()
