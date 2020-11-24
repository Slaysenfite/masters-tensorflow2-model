from random import uniform, seed

import numpy as np
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import Precision, Recall

from training_loops.OptimizerHelper import get_trainable_weights, set_trainable_weights

seed(1)

INERTIA = 0.75
C1 = 2
C2 = 2
V_MAX = 2


class Particle:
    def __init__(self, position, fitness, velocity=0.0):
        self.position = position
        self.velocity = velocity
        self.gbest = None
        self.pbest = position
        self.current_fitness = fitness
        self.best_fitness = fitness
        self.gbest_fitness = None


class PsoEnv():
    def __init__(self, iterations=5, swarm_size=8, model=None, X=None, y=None, layers_to_optimize=(Conv2D, Dense)):
        self.iterations = iterations
        self.swarm_size = swarm_size
        self.model = model
        self.X = X
        self.y = y
        self.layers_to_optimize = layers_to_optimize

    def get_pso_model(self):
        iteration = 0;
        loss_metric = CategoricalCrossentropy(from_logits=True)
        weights = get_trainable_weights(self.model, self.layers_to_optimize)

        swarm = self.initialize_swarm(self.swarm_size, weights, self.model, loss_metric, self.X, self.y)

        best_particle = self.find_best_particle(swarm)
        self.set_gbest(swarm, best_particle)

        while iteration < self.iterations:
            self.update_positions(swarm, self.model, loss_metric, self.X, self.y)

            self.update_gbest(swarm)

            print(' PSO training for iteration {}'.format(iteration + 1) + ' - Best fitness of {}'.format(
                swarm[0].gbest_fitness))
            iteration += 1
        best_weights = swarm[0].gbest

        swarm = None

        return set_trainable_weights(self.model, best_weights, self.layers_to_optimize)

    def initialize_swarm(self, swarm_size, weights, model, loss_metric, X, y):
        particles = [None] * swarm_size
        particles[0] = Particle(weights, self.calc_position_fitness(weights, model, loss_metric, X, y))
        for p in range(1, swarm_size):
            new_weights = [w * uniform(-1, 1) for w in weights]
            initial_fitness = self.calc_position_fitness(new_weights, model, loss_metric, X, y)
            particles[p] = Particle(np.array(new_weights), initial_fitness)
        return particles

    def calc_position_fitness(self, weights, model, loss_metric, X, y):
        set_trainable_weights(model, weights, self.layers_to_optimize)
        precision_metric = Precision()
        recall_metric = Recall()
        ŷ = model(X, training=True)
        loss = loss_metric(y, ŷ).numpy()
        precision = precision_metric(y, ŷ).numpy()
        recall = recall_metric(y, ŷ).numpy()
        #figute out why precision and recall don't calculate correctly
        return (1 - precision) + (1 - recall)


    def set_gbest(self, particles, best_particle):
        for particle in particles:
            particle.gbest = best_particle.position
            particle.gbest_fitness = best_particle.best_fitness

    def update_positions(self, particles, model, loss_metric, X, y):
        for particle in particles:
            particle.velocity = self.update_velocity(particle, INERTIA, [C1, C2])
            particle.position = self.calc_new_position(particle)
            particle.current_fitness = self.calc_position_fitness(particle.position, model, loss_metric, X, y)
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

    def find_best_particle(self, particles):
        best_particle = particles[0]
        for i in range(1, len(particles)):
            if particles[i].current_fitness < best_particle.current_fitness:
                best_particle = particles[i]
        return best_particle

    def update_velocity(self, particle, inertia_weight, acc_c):
        # TODO: Look into clamping the velocity
        initial = (inertia_weight) * (particle.velocity)
        cognitive_component = (acc_c[0]) * (uniform(0, 1)) * (particle.pbest - particle.position)
        social_component = (acc_c[1]) * (uniform(0, 1)) * (particle.gbest - particle.position)

        return initial + cognitive_component + social_component

    def calc_new_position(self, particle):
        return particle.position + particle.velocity
