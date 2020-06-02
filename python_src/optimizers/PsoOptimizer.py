"""
    Possible features needed
    1. Toggle between Gbest and Lbest
    2. Specify max number of generations
"""
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Optimizer
from tensorflow.python.ops import state_ops


class PSO(Optimizer):

    def __init__(self, num_particles, max_gens, topology='lbest', **kwargs):
        super(PSO, self).__init__(**kwargs)
        self.max_gens = max_gens
        self.topology = topology
        self.num_particles - num_particles
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    def get_updates(self, loss, params):
        self.updates = [state_ops.assign_add(self.iterations, 1)]
        return self.updates

    def init_population(self):
        population = []
        for i in range(self.num_particles):
            # append new particle
            population.append()


