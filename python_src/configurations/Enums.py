from enum import Enum


class LearningOptimization(Enum):
    ADAM = "Adam"


class MetaHeuristic(Enum):
    NONE = "NONE"
    PSO = "Particle swarm optimization"
    GA = "Genetic Algorithm"

class MetaHeuristicOrder(Enum):
    NONE = "N/A"
    FIRST = "First"
    LAST = "Last"
