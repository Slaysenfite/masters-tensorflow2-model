from src.model.Enums import ActivationFunctions, LearningOptimization, Pooling


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


def create_hyperparameter_singleton():
    return Hyperparameters(
        10,
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
