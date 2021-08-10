from random import uniform, seed, randint

from tensorflow._api.v2 import math
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense

from training_loops.OptimizerHelper import get_trainable_weights, set_trainable_weights, calc_solution_fitness, \
    determine_loss_function_based_on_fitness_function, convert_tenor_weights_to_tf_variable

HALL_OF_FAME_SIZE = 3

CROSSOVER_THRESHOLD = 60

TOURNAMENT_SIZE = 4

CROSSOVER_PROBABILITY = 50

MUTATION_PROBABILTY = 3

seed(1)


class Solution:
    def __init__(self, weights, fitness):
        self.weights = weights
        self.fitness = fitness
        # self.weights_flat, self.shapes = flatten(weights_arr, list(), list())
        # self.shape = weights_arr.shape


class GaEnv():
    def __init__(self, fitness_function=calc_solution_fitness, iterations=10, population_size=30, model=None, X=None, y=None,
                 layers_to_optimize=(Conv2D, Dense)):
        self.iterations = iterations
        self.population_size = population_size
        self.model = model
        self.X = X
        self.y = y
        self.layers_to_optimize = layers_to_optimize
        self.fitness_function = fitness_function
        self.loss_metric = determine_loss_function_based_on_fitness_function(self.fitness_function)

    def get_ga_model(self):
        iteration = 0
        weights = get_trainable_weights(self.model, self.layers_to_optimize)

        individuals = self.initialize_population(self.population_size, weights, self.model, self.X, self.y)
        individuals = self.update_fitness(individuals, self.model, self.X, self.y)

        while iteration < self.iterations:
            individuals = self.reproduce_next_gen(individuals)

            individuals = self.update_fitness(individuals, self.model, self.X, self.y)
            print('\n GA training for iteration {}'.format(iteration + 1) + ' - Best fitness of {}'.format(
                individuals[0].fitness))
            iteration += 1
        best_weights = convert_tenor_weights_to_tf_variable(individuals[0].weights)

        return set_trainable_weights(self.model, best_weights, self.layers_to_optimize)

    def initialize_population(self, population_size, weights, model, X, y):
        individuals = [None] * population_size
        weights = [[w * 1 for w in weight] for weight in weights]
        fitness = self.fitness_function(weights, model, self.loss_metric, X, y)
        individuals[0] = Solution(weights, fitness)
        for p in range(1, population_size):
            new_weights = [[w * uniform(0, 1) for w in weight] for weight in weights]
            individuals[p] = Solution(new_weights, 1000)
        return individuals

    def find_best_individual(self, individuals):
        best_individual = individuals[0]
        for i in range(1, len(individuals)):
            if individuals[i].fitness < best_individual.fitness:
                best_individual = individuals[i]
        return best_individual

    def update_fitness(self, individuals, model, X, y):
        for individual in individuals:
            individual.fitness = self.fitness_function(individual.weights, model, self.loss_metric, X, y)
        individuals.sort(key=lambda indv: indv.fitness, reverse=False)
        return individuals

    def update_weight_arr(self, individuals):
        for individual in individuals:
            individual.weight_arr = individual.w

    def reproduce_next_gen(self, individuals):
        next_gen = list()
        for i in range(HALL_OF_FAME_SIZE):
            next_gen.append(individuals[i])
        while len(next_gen) != self.population_size:
            indv1, indv2 = self.select_indvs_for_reproduction(individuals)
            rand = randint(1, 100)
            if rand > CROSSOVER_THRESHOLD:
                offspring = self.two_point_crossover(indv1, indv2)
                next_gen.append(self.mutate(offspring))
            else:
                if next_gen.count(indv1) == 0:
                    next_gen.append(indv1)
        return next_gen

    def select_indvs_for_reproduction(self, individuals, tournament_size=TOURNAMENT_SIZE):
        tournament = list()
        indices = self.gen_rand_num_list(tournament_size, 0, len(individuals) - 1);
        for i in range(tournament_size):
            index = indices[i]
            tournament.append(individuals[index])
        tournament.sort(key=lambda indv: indv.fitness, reverse=False)
        return tournament[0], tournament[1]

    def two_point_crossover(self, individual1, individual2):
        one = individual1.weights
        two = individual2.weights

        new_weights = []
        for i in range(len(one)):
            vars = []
            for n in range(len(one[i])):
                vars.append(self.perform_element_level_crossover(one[i][n], two[i][n]))
            new_weights.append(
                vars
            )

        return Solution(new_weights,
                        self.fitness_function(new_weights, self.model, self.loss_metric, self.X, self.y))

    @staticmethod
    def perform_element_level_crossover(one_element, two_element):
        rand_num = uniform(0, 100)
        if rand_num > CROSSOVER_PROBABILITY:
            return two_element
        else:
            return one_element

    def mutate(self, offspring):
        new_weights = []
        for i in range(len(offspring.weights)):
            vars = []
            for n in range(len(offspring.weights[i])):
                if randint(1, 100) <= MUTATION_PROBABILTY:
                    new_vals = math.scalar_mul(uniform(-1, 1), offspring.weights[i][n])
                    vars.append(new_vals)
                else:
                    vars.append(offspring.weights[i][n])
            new_weights.append(
                vars
            )

        return Solution(new_weights,
                        self.fitness_function(new_weights, self.model, self.loss_metric, self.X, self.y))

    def gen_rand_num_list(self, size, a, b):
        l = list()
        for i in range(size):
            l.append(randint(a, b))
        return l