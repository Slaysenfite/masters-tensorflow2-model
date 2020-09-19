from model.Enums import LearningOptimization


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


def create_hyperparameter_singleton():
    return Hyperparameters(
        75,
        1e-3,
        32,
        LearningOptimization.ADAM,
        0.3
    )


hyperparameters = create_hyperparameter_singleton()
