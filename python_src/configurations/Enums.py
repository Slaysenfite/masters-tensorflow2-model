from enum import Enum


class LearningOptimization(Enum):
    SGD = "Stochastic gradient descent"
    ADAM = "Adam"
    PSO = "Particle swarm optimization"
    HYBRID = "PSO and Gradient Descent Hybrid"


class ActivationFunctions(Enum):
    RELU = "Rectified linear unit activation"
    SIGMOID = "Sigmoid activation"
    SOFTMAX = "softmax"


class Pooling(Enum):
    TWO_BY_TWO = (2, 2)
