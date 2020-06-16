"""
    Possible features needed
    1. Toggle between Gbest and Lbest
    2. Specify max number of generations
"""
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.ops import state_ops


class PsoOptimizer(OptimizerV2):

    def __init__(self, num_particles=25, max_gens=100, topology='lbest', inertia_weight=0.8, c1=2, c2=2, vmax=2,
                 **kwargs):
        super(PsoOptimizer, self).__init__("PSO", **kwargs)
        self.max_gens = max_gens
        self.topology = topology
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.c1 = c1
        self.c2 = c2
        self.vmax = vmax

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

    def calculate_loss(self):
        weights = self.get_weights()

    def calculate_solution_score(self, position):
        raise NotImplementedError("")

    def update(self):
        raise NotImplementedError("")
