from enum import Enum


class LearningOptimization(Enum):
    SGD = "Stochastic gradient descent"
    ADAM = "Adam"
    PSO = "Particle swarm optimization"
    GA = "genetic algorithm"


class ActivationFunctions(Enum):
    RELU = "Rectified linear unit activation"
    SIGMOID = "Sigmoid activation"
