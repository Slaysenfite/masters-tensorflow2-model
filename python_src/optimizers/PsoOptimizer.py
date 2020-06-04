"""
    Possible features needed
    1. Toggle between Gbest and Lbest
    2. Specify max number of generations
"""
from tensorflow import Tensor
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops



class PsoOptimizer(OptimizerV2):

    def __init__(self, num_particles=100, max_gens=100, topology='lbest', name="PSO", **kwargs):
        super(PsoOptimizer, self).__init__(name, **kwargs)
        self.max_gens = max_gens
        self.topology = topology
        self.num_particles = num_particles
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    def _resource_apply_dense(self, grad, var):
        return [state_ops.assign_add(self.iterations, 1)]

    def get_config(self):
        config = super(PsoOptimizer, self).get_config()
        config.update({
            "num_particles": self._serialize_hyperparameter("num_particles"),
            "max_gens": self._serialize_hyperparameter("max_gens"),
            "topology": self._serialize_hyperparameter("topology"),
        })
        return config

    def _create_slots(self, var_list):
        if self.num_particles:
            for var in var_list:
                self.add_slot(var, "num_particles")
        if self.max_gens:
            for var in var_list:
                self.add_slot(var, "max_gens")
        if self.topology:
            for var in var_list:
                self.add_slot(var, "topology")

    """
    - resource_apply_dense (update variable given gradient tensor is dense)
    - resource_apply_sparse (update variable given gradient tensor is sparse)
    - create_slots (if your optimizer algorithm requires additional variables)
    - get_config (serialization of the optimizer, include all hyper parameters)
    """

