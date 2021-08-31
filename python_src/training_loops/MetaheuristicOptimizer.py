from abc import abstractmethod, ABC

from tensorflow.python.keras.layers import Conv2D, Dense

from training_loops.OptimizerHelper import calc_solution_fitness, determine_loss_function_based_on_fitness_function


class MetaheuristicOptimizer(ABC):
    def __init__(self,
                 fitness_function=calc_solution_fitness,
                 iterations=10,
                 num_solutions=30,
                 model=None,
                 X=None,
                 y=None,
                 layers_to_optimize=(Conv2D, Dense),
                 num_layers=5):
        self.iterations = iterations
        self.num_solutions = num_solutions
        self.model = model
        self.X = X
        self.y = y
        self.layers_to_optimize = layers_to_optimize
        self.num_layers = num_layers
        self.fitness_function = fitness_function
        self.loss_metric = determine_loss_function_based_on_fitness_function(self.fitness_function)
        super().__init__()

    @abstractmethod
    def get_optimized_model(self):
        pass
