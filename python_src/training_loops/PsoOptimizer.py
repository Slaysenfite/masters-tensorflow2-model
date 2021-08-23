from random import uniform, seed

from tensorflow._api.v2 import math
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense

from training_loops.MetaheuristicOptimizer import MetaheuristicOptimizer
from training_loops.OptimizerHelper import get_trainable_weights, set_trainable_weights, calc_solution_fitness, \
    convert_tenor_weights_to_tf_variable, perform_tensor_operations, add_three_tensors

seed(1)

INERTIA = 0.75
C1 = 2
C2 = 2
V_MAX = 2


class Particle:
    def __init__(self, position, fitness, velocity):
        self.position = position
        self.velocity = velocity
        self.gbest = None
        self.pbest = position
        self.current_fitness = fitness
        self.best_fitness = fitness
        self.gbest_fitness = None


class PsoEnv(MetaheuristicOptimizer):
    def __init__(self, fitness_function=calc_solution_fitness, iterations=5, num_solutions=8, model=None, X=None,
                 y=None, layers_to_optimize=(Conv2D, Dense)):
        super().__init__(fitness_function, iterations, num_solutions, model, X, y, layers_to_optimize)

    def get_optimized_model(self):
        print('\nRunning PSO algorithm')
        iteration = 0
        weights = get_trainable_weights(model=self.model, keras_layers=self.layers_to_optimize,
                                        num_layers=self.num_layers)

        swarm = self.initialize_swarm(self.num_solutions, weights, self.model, self.loss_metric, self.X, self.y)

        best_particle = self.find_best_particle(swarm)
        self.set_gbest(swarm, best_particle)

        while iteration < self.iterations:
            self.update_positions(swarm, self.model, self.loss_metric, self.X, self.y)

            self.update_gbest(swarm)

            print(' PSO training for iteration {} - Best fitness of {}'.format((iteration + 1), swarm[0].gbest_fitness))
            iteration += 1
        best_weights = convert_tenor_weights_to_tf_variable(swarm[0].gbest)

        return set_trainable_weights(model=self.model, weights=best_weights, keras_layers=self.layers_to_optimize,
                                     num_layers=self.num_layers)

    def initialize_swarm(self, swarm_size, weights, model, loss_metric, X, y):
        particles = [None] * swarm_size
        starting_velocity = [[w * 0 for w in weight] for weight in weights]
        particles[0] = Particle(weights, self.fitness_function(weights, model, loss_metric, X, y, self.num_layers), starting_velocity)
        print(' PSO starting fitness of {}'.format(particles[0].gbest_fitness))
        for p in range(1, swarm_size):
            new_weights = [[w * uniform(0, 1) for w in weight] for weight in weights]
            initial_fitness = self.fitness_function(new_weights, model, loss_metric, X, y, self.num_layers)
            particles[p] = Particle(position=new_weights, fitness=initial_fitness, velocity=starting_velocity)
        return particles

    @staticmethod
    def set_gbest(particles, best_particle):
        for particle in particles:
            particle.gbest = best_particle.position
            particle.gbest_fitness = best_particle.best_fitness

    def update_positions(self, particles, model, loss_metric, X, y):
        for particle in particles:
            particle.velocity = self.update_velocity(particle, INERTIA, [C1, C2])
            particle.position = self.calc_new_position(particle)
            particle.current_fitness = self.fitness_function(particle.position, model, loss_metric, X, y, self.num_layers)
            if particle.current_fitness < particle.best_fitness:
                particle.pbest = particle.position
                particle.best_fitness = particle.current_fitness

    def update_gbest(self, swarm):
        best_particle = swarm[0]
        initial_best_fitness = best_particle.gbest_fitness
        for i in range(1, len(swarm)):
            if swarm[i].best_fitness < best_particle.gbest_fitness:
                best_particle = swarm[i]
        if best_particle.best_fitness < initial_best_fitness:
            self.set_gbest(swarm, best_particle)

    @staticmethod
    def find_best_particle(particles):
        best_particle = particles[0]
        for i in range(1, len(particles)):
            if particles[i].current_fitness < best_particle.current_fitness:
                best_particle = particles[i]
        return best_particle

    def update_velocity(self, particle, inertia_weight, acc_c):
        initial = [[inertia_weight * v for v in velocity] for velocity in particle.velocity]
        cognitive_component = self.get_cognitive_component(particle, acc_c)
        social_component = self.get_social_component(particle, acc_c)

        return add_three_tensors(initial, cognitive_component, social_component)

    @staticmethod
    def get_cognitive_component(particle, acc_c):
        magic = (acc_c[0]) * (uniform(0, 1))
        cognitive_component = perform_tensor_operations(math.subtract, particle.pbest, particle.position)
        cognitive_component = [[c * magic for c in component] for component in cognitive_component]
        return cognitive_component

    @staticmethod
    def get_social_component(particle, acc_c):
        magic = (acc_c[1]) * (uniform(0, 1))
        social_component = perform_tensor_operations(math.subtract, particle.gbest, particle.position)
        social_component = [[c * magic for c in component] for component in social_component]
        return social_component

    @staticmethod
    def calc_new_position(particle):
        return perform_tensor_operations(math.add, particle.position, particle.velocity)
