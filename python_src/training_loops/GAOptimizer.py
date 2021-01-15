from random import uniform, seed, randint

import numpy as np
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import CategoricalAccuracy, TrueNegatives, TruePositives, \
    FalsePositives, FalseNegatives

from training_loops.OptimizerHelper import get_trainable_weights, set_trainable_weights

TOURNAMENT_SIZE = 12

MUTATION_PROBABILITY = 1

CROSSOVER_PROBABILITY = 80

seed(1)


class Solution:
    def __init__(self, weights_arr, fitness):
        self.weights_arr = weights_arr
        self.fitness = fitness
        self.weights_flat, self.shapes = flatten(weights_arr, list(), list())
        print('poo')
        # self.reassembled = reconstruct(self.weights_flat, self.shapes, list())
        #
        # isEqual = self.reassembled == self.weights_arr
        # print(isEqual)


class GaEnv():
    def __init__(self, iterations=100, population_size=100, model=None, X=None, y=None,
                 layers_to_optimize=(Conv2D, Dense)):
        self.iterations = iterations
        self.population_size = population_size
        self.model = model
        self.X = X
        self.y = y
        self.layers_to_optimize = layers_to_optimize

    def get_ga_model(self):
        iteration = 0;
        loss_metric = CategoricalCrossentropy()
        weights = get_trainable_weights(self.model, self.layers_to_optimize)

        individuals = self.initialize_population(self.population_size, weights, self.model, loss_metric, self.X, self.y)
        self.update_fitness(individuals, self.model, loss_metric, self.X, self.y)

        while iteration < self.iterations:
            best_individual = self.find_best_individual(individuals)
            self.reproduce_next_gen(individuals, best_individual)

            print(' GA training for iteration {}'.format(iteration + 1) + ' - Best fitness of {}'.format(
                best_individual.fitness))
            iteration += 1
        best_weights = best_individual.weights_arr

        return set_trainable_weights(self.model, best_weights, self.layers_to_optimize)

    def initialize_population(self, population_size, weights, model, loss_metric, X, y):
        individuals = [None] * population_size
        individuals[0] = Solution(weights, self.calc_fitness(weights, model, loss_metric, X, y))
        for p in range(1, population_size):
            new_weights = [w * uniform(0, 0) for w in weights]
            initial_fitness = self.calc_fitness(new_weights, model, loss_metric, X, y)
            individuals[p] = Solution(np.array(new_weights), initial_fitness)
        return individuals

    def calc_fitness(self, weights, model, loss_metric, X, y):
        set_trainable_weights(model, weights, self.layers_to_optimize)
        ŷ = model(X, training=True)

        accuracy_metric = CategoricalAccuracy()
        tn = TrueNegatives()
        tp = TruePositives()
        fp = FalsePositives()
        fn = FalseNegatives()

        accuracy = accuracy_metric(y, ŷ).numpy()
        loss = loss_metric(y, ŷ).numpy()
        tn_score = tn(y, ŷ).numpy()
        tp_score = tp(y, ŷ).numpy()
        fp_score = fp(y, ŷ).numpy()
        fn_score = fn(y, ŷ).numpy()

        precision = tp_score / (tp_score + fn_score)
        specificity = tn_score / (tn_score + fp_score)

        fpr = fp_score / (tn_score + fn_score + tp_score + fp_score)
        return (fpr) + (2 * loss) + (1 - specificity) + (1 - accuracy)

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

    def reproduce_next_gen(self, individuals, best_individual):
        next_gen = list()
        next_gen.append(best_individual)
        while len(next_gen) != len(individuals):
            indv1, indv2 = self.select_indvs_for_reproduction(individuals)
            offspring = self.two_point_crossover(indv1, indv2, best_individual.shape)
            next_gen.append(offspring)
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
        offspring = np.empty_like(individual1.weights_flat)

        one = individual1.weights_flat
        two = individual2.weights_flat

        for i in range(len(one)):
            rand_num = uniform(0, 100)
            if rand_num > CROSSOVER_PROBABILITY:
                offspring[i] = two[i]
            else:
                offspring[i] = one[i]

        self.mutate(offspring)

        new_weight = offspring.reshape(shape)

        return Solution(new_weight,
                        self.calc_fitness(new_weight, self.model, CategoricalCrossentropy(), self.X, self.y))

        #
        # p1 = randint(0, len(one) - 1)
        # p2 = randint(0, len(two) - 1)
        #
        # while p1 == p2:
        #     p2 = randint(0, len(two) - 1)
        # if p1 < p2:
        #     temp = p2
        #     p2 = p1
        #     p1 = temp
        #
        # if p1 >= 0:
        #     offspring[0:p1] = one[0:p1]
        # if p2 - p1 >= 0:
        #     offspring[p1:p2] = two[p1:p2]

        return

    def mutate(self, offspring):
        for i in range(len(offspring)):
            rand_num = uniform(0, 100)
            if rand_num > MUTATION_PROBABILITY:
                offspring[i] = offspring[i] * uniform(0, 1)

    def gen_rand_num_list(self, size, a, b):
        l = list()
        for i in range(size):
            l.append(randint(a, b))
        return l


def get_trainable_weights(model, keras_layers=(Dense, Conv2D), ):
    weights = []
    shapes = []
    for layer in model.layers:
        if (layer.trainable != True or len(layer.trainable_weights) == 0 or layer.name == 'predictions'):
            pass
        if isinstance(layer, keras_layers):
            weights.append(layer.trainable_weights)
    return weights


def set_trainable_weights(model, weights, keras_layers=(Dense, Conv2D)):
    i = 0
    for layer in model.layers:
        if (layer.trainable != True or len(layer.weights) == 0 or layer.name == 'predictions'):
            pass
        if isinstance(layer, keras_layers):
            layer._trainable_weights = weights[i]
            i += 1
    return model


def flatten(arr, weights, shapes):
    for i in range(len(arr)):
        for c in range(len(arr[i])):
            weights.append(arr[i][c])
        shapes.append(len(arr[i]))
    return np.array(weights).flatten(), shapes
#
#
# def reconstruct(weights_flat, shapes, reconstructed):
#     index = 0
#     for shape in shapes:
#         arr = list()
#         for s in shape:
#             if s is not None:
#                 arr.append(weights_flat[index])
#                 reconstructed.append(np.array(arr))
#                 index += 1
#     print(shape)
