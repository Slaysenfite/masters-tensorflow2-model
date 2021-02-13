from random import uniform, seed, randint

import numpy as np
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import CategoricalAccuracy, TrueNegatives, TruePositives, \
    FalsePositives, FalseNegatives

from training_loops.OptimizerHelper import get_trainable_weights, set_trainable_weights

CROSSOVER_THRESHOLD = 85

TOURNAMENT_SIZE = 24

CROSSOVER_PROBABILITY = 30

CULLING_SIZE = 10

seed(1)


class Solution:
    def __init__(self, weights_arr, fitness):
        self.weights_arr = weights_arr
        self.fitness = fitness
        self.weights_flat, self.shapes = flatten(weights_arr, list(), list())
        self.shape = weights_arr.shape


class GaEnv():
    def __init__(self, iterations=10, population_size=30, model=None, X=None, y=None,
                 layers_to_optimize=(Conv2D, Dense)):
        self.iterations = iterations
        self.population_size = population_size
        self.model = model
        self.X = X
        self.y = y
        self.layers_to_optimize = layers_to_optimize

    def get_ga_model(self):
        iteration = 0
        loss_metric = CategoricalCrossentropy()
        weights = get_trainable_weights(self.model, self.layers_to_optimize)

        individuals = self.initialize_population(self.population_size, weights, self.model, loss_metric, self.X, self.y)

        while iteration < self.iterations:
            individuals = self.reproduce_next_gen(individuals)

            print(' GA training for iteration {}'.format(iteration + 1) + ' - Best fitness of {}'.format(
                individuals[0].fitness))
            iteration += 1
        best_weights = individuals[0].weights_arr

        return set_trainable_weights(self.model, best_weights, self.layers_to_optimize)

    def initialize_population(self, population_size, weights, model, loss_metric, X, y):
        individuals = [None] * population_size
        individuals[0] = Solution(weights, self.calc_fitness(weights, model, loss_metric, X, y))
        for p in range(1, population_size):
            new_weights = [w * uniform(-2, 1) for w in weights]
            initial_fitness = self.calc_fitness(new_weights, model, loss_metric, X, y)
            individuals[p] = Solution(np.array(new_weights), initial_fitness)
        return individuals

    def calc_fitness(self, weights, model, loss_metric, X, y):
        set_trainable_weights(model, weights, self.layers_to_optimize)
        ŷ = model(X, training=True)
        loss = loss_metric(y, ŷ).numpy()

        return loss

    def find_best_individual(self, individuals):
        best_individual = individuals[0]
        for i in range(1, len(individuals)):
            if individuals[i].fitness < best_individual.fitness:
                best_individual = individuals[i]
        return best_individual

    def update_fitness(self, individuals, model, loss_metric, X, y):
        for individual in individuals:
            individual.fitness = self.calc_fitness(individual.weights_arr, model, loss_metric, X, y)

    def update_weight_arr(self, individuals):
        for individual in individuals:
            individual.weight_arr = individual.w

    def reproduce_next_gen(self, individuals):
        individuals.sort(key=lambda indv: indv.fitness, reverse=False)
        shape = individuals[0].shape
        next_gen = list()
        for i in range(len(individuals) - CULLING_SIZE):
            next_gen.append(individuals[i])
        while len(next_gen) != len(individuals):
            indv1, indv2 = self.select_indvs_for_reproduction(individuals)
            rand = randint(1, 100)
            if rand > CROSSOVER_THRESHOLD:
                offspring = self.two_point_crossover(indv1, indv2, shape)
                next_gen.append(offspring)
            else:
                next_gen.append(self.mutate(indv1.weights_flat, shape))
        next_gen.sort(key=lambda indv: indv.fitness, reverse=False)
        return next_gen

    def select_indvs_for_reproduction(self, individuals, tournament_size=TOURNAMENT_SIZE):
        tournament = list()
        indices = self.gen_rand_num_list(tournament_size, 0, len(individuals) - 1);
        for i in range(tournament_size):
            index = indices[i]
            tournament.append(individuals[index])
        tournament.sort(key=lambda indv: indv.fitness, reverse=False)
        return tournament[0], tournament[1]

    def two_point_crossover(self, individual1, individual2, shape):
        offspring = list()

        one = individual1.weights_flat.tolist()
        two = individual2.weights_flat.tolist()

        for r in range(len(one)):
            offspring.append(np.zeros_like(one[r]))
            for c in range(len(one[r])):
                if isinstance(offspring[r][c], np.float32):
                    offspring[r][c] = self.perform_element_level_crossover(one[r][c], two[r][c])
                else:
                    for w in range(len(one[r][c])):
                        offspring[r][c][w] = self.perform_element_level_crossover(one[r][c][w], two[r][c][w])

        new_weight = np.array(offspring).reshape(shape)

        return Solution(new_weight,
                        self.calc_fitness(new_weight, self.model, CategoricalCrossentropy(), self.X, self.y))

    def perform_element_level_crossover(self, one_element, two_element):
        rand_num = uniform(0, 100)
        if rand_num > CROSSOVER_PROBABILITY:
            return two_element
        else:
            return one_element

    def mutate(self, offspring, shape=(2, 2)):
        for r in range(len(offspring)):
            for c in range(len(offspring[r])):
                if isinstance(offspring[r][c], np.float32):
                    if offspring[r][c] == 0:
                        offspring[r][c] = uniform(-1, 1)
                    else:
                        offspring[r][c] *= uniform(0, 1)
                else:
                    new_vals = [w * uniform(0, 1) for w in offspring[r][c]]
                    offspring[r][c] = new_vals

        new_weight = np.array(offspring).reshape(shape)
        return Solution(new_weight,
                        self.calc_fitness(new_weight, self.model, CategoricalCrossentropy(), self.X, self.y))

    def gen_rand_num_list(self, size, a, b):
        l = list()
        for i in range(size):
            l.append(randint(a, b))
        return l


def flatten(arr, weights, shapes):
    for i in range(len(arr)):
        for c in range(len(arr[i])):
            weights.append(arr[i][c])
        shapes.append(len(arr[i]))
    return np.array(weights).flatten(), shapes