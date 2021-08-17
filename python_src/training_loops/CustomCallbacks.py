import logging as logging

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback

from training_loops.GAOptimizer import GaEnv
from training_loops.OptimizerHelper import calc_solution_fitness
from training_loops.PsoOptimizer import PsoEnv


class RunMetaHeuristicOnPlateau(Callback):
    """Run meta heuristic when a metric has stopped improving.

    This callback monitors a quantity and if no improvement is
    seen for a 'patience' number of epochs, the a meta-heuristic
    algorithm is run

    Example:

    Arguments:
        monitor: quantity to be monitored.
        patience: number of epochs with no improvement after which learning rate
          will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode,
          the learning rate will be reduced when the
          quantity monitored has stopped decreasing; in `'max'` mode it will be
          reduced when the quantity monitored has stopped increasing; in `'auto'`
          mode, the direction is automatically inferred from the name of the
          monitored quantity.
        min_delta: threshold for measuring the new optimum, to only focus on
          significant changes.
        cooldown: number of epochs to wait before resuming normal operation after
          algorithm has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self,
                 X,
                 y,
                 monitor='val_loss',
                 patience=5,
                 verbose=1,
                 mode='auto',
                 min_delta=0.0001,
                 cooldown=0,
                 meta_heuristic='pso',
                 fitness_function=calc_solution_fitness,
                 population_size=30,
                 iterations=10,
                 **kwargs):
        super(RunMetaHeuristicOnPlateau, self).__init__()

        self.monitor = monitor
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            logging.warning('`epsilon` argument is deprecated and '
                            'will be removed, use `min_delta` instead.')
        self.X = X
        self.y = y
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.meta_heuristic = meta_heuristic
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.iterations = iterations
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning('Run Metaheuristic On Plateau Reducing mode %s is unknown, '
                            'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Run metaheuristic on plateau conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    if self.meta_heuristic == 'pso':
                        meta_opt = self.apply_swarm_optimization()
                    elif self.meta_heuristic == 'ga':
                        meta_opt = self.apply_genetic_algorithm()
                    else:
                        raise Exception('[WARNING] Incorrect meta-heuristic specified, must be either "pso" or "ga"')
                    if self.verbose > 0:
                        print('\nEpoch %05d: RunMetaHeuristicOnPlateau running meta-heuristic algorithm %s with '
                              'hyperparameters [population size: %d iterations: %d]'
                              % (epoch + 1, self.meta_heuristic, self.population_size, self.iterations))
                    model = meta_opt.get_optimized_model()
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
                    return model

    def in_cooldown(self):
        return self.cooldown_counter > 0

    def apply_swarm_optimization(self):
        return PsoEnv(num_solutions=self.population_size,
                      iterations=self.iterations,
                      model=self.model,
                      X=self.X,
                      y=self.y,
                      fitness_function=self.fitness_function)

    def apply_genetic_algorithm(self):
        return GaEnv(num_solutions=self.population_size,
                     iterations=self.iterations,
                     model=self.model,
                     X=self.X,
                     y=self.y,
                     fitness_function=self.fitness_function)
