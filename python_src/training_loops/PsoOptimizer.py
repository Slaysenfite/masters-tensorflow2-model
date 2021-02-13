from random import uniform, seed

import numpy as np
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.python.keras.metrics import CategoricalAccuracy, TrueNegatives, TruePositives, \
    FalsePositives, FalseNegatives

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
        loss_metric = CategoricalCrossentropy()
        self.model.reset_metrics()
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

        return set_trainable_weights(self.model, best_weights, self.layers_to_optimize)

    def initialize_swarm(self, swarm_size, weights, model, loss_metric, X, y):
        particles = [None] * swarm_size
        particles[0] = Particle(weights, self.calc_position_fitness(weights, model, loss_metric, X, y))
        for p in range(1, swarm_size):
            new_weights = [w * uniform(0, 1) for w in weights]
            initial_fitness = self.calc_position_fitness(new_weights, model, loss_metric, X, y)
            particles[p] = Particle(np.array(new_weights), initial_fitness)
        return particles

    def calc_position_fitness(self, weights, model, loss_metric, X, y):
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

        # precision = tp_score / (tp_score + fn_score)
        specificity = tn_score / (tn_score + fp_score)


        fpr = fp_score / (tn_score + fn_score + tp_score + fp_score)
        return (fpr) + (2*loss) + (1 - specificity)+(1-accuracy)

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
        initial = (inertia_weight) * (particle.velocity)
        cognitive_component = (acc_c[0]) * (uniform(0, 1)) * (particle.pbest - particle.position)
        social_component = (acc_c[1]) * (uniform(0, 1)) * (particle.gbest - particle.position)

        new_velocity = initial + cognitive_component + social_component

        # self.clamp_velocity(new_velocity)

        return new_velocity

    def calc_new_position(self, particle):
        return particle.position + particle.velocity

    def clamp_velocity(self, new_velocity):
        for v in new_velocity.flat:
            for v1 in v.flat:
                if v1 < -V_MAX:
                    v1 = -V_MAX
                if v1 > V_MAX:
                    v1 = V_MAX
