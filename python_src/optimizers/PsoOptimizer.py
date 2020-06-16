from random import uniform, seed, randint, random

import numpy as np
from tensorflow.python.keras.layers import Conv2D, Dense

seed(1)
rand_seed = randint(0, 10)
seed(rand_seed)

INERTIA = 0.75
C1 = 2
C2 = 2
V_MAX = 2


class Particle:
    def __init__(self, position, loss, velocity=0.0):
        self.position = position
        self.velocity = velocity
        self.gbest = None
        self.pbest = position
        self.current_loss = loss
        self.best_loss = loss


class PsoEnv():
    def __init__(self, iterations, swarm_size, model, loss_metric, X, y):
        self.iterations = iterations
        self.swarm_size = swarm_size
        self.model = model
        self.loss_metric = loss_metric
        self.X = X
        self.y = y

    def get_pso_model(self):
        iteration = 0;
        weights = self.get_trainable_weights(self.model)

        swarm = self.initialize_swarm(self.swarm_size, weights, self.model, self.loss_metric, self.X, self.y)
        while iteration < self.iterations:
            print('PSO training for iteration {}'.format(iteration))
            self.update_gbest(swarm)
            self.update_positions(swarm, self.model, self.loss_metric, self.X, self.y)
            iteration += 1
        best_weights = self.find_best_particle(swarm).position
        return self.set_trainable_weights(self.model, best_weights)

    def initialize_swarm(self, swarm_size, weights, model, loss_metric, X, y):
        particles = [None] * swarm_size
        particles[0] = Particle(weights, self.calc_position_loss(weights, model, loss_metric, X, y))
        for p in range(1, swarm_size):
            new_weights = [w * uniform(0, 1) for w in weights]
            initial_loss = self.calc_position_loss(new_weights, model, loss_metric, X, y)
            particles[p] = Particle(np.array(new_weights), initial_loss)
        return particles

    def calc_position_loss(self, weights, model, loss_metric, X, y):
        self.set_trainable_weights(model, weights)
        ŷ = model(X, training=True)
        loss_metric(y, ŷ)
        return loss_metric.result().numpy()

    def update_gbest(self, particles):
        best_particle = self.find_best_particle(particles)
        for particle in particles:
            particle.gbest = best_particle.position

    def update_positions(self, particles, model, loss_metric, X, y):
        for particle in particles:
            self.update_velocity(particle, INERTIA, [C1, C2])
            particle.position = self.calc_new_position(particle)
            particle.current_loss = self.calc_position_loss(particle.position, model, loss_metric, X, y)
            if particle.current_loss < particle.best_loss:
                particle.pbest = particle.position
                particle.best_loss = particle.current_loss

    def get_trainable_weights(self, model):
        weights = []
        for layer in model.layers:
            if (layer.trainable != True or len(layer.trainable_weights) == 0):
                pass
            if isinstance(layer, (Conv2D, Dense)):
                t_weights_for_layer = []
                t_weights_for_layer.append(layer.weights[0].numpy())
                weights.append(t_weights_for_layer)
        return np.array(weights)

    def set_trainable_weights(self, model, weights):
        i = 0
        for layer in model.layers:
            if (layer.trainable != True or len(layer.weights) == 0):
                pass
            if isinstance(layer, (Conv2D, Dense)):
                layer.weights[0] = weights[i]
                i += 1
        return model

    def find_best_particle(self, particles):
        best_particle = particles[0]
        for i in range(0, len(particles)):
            if particles[i].current_loss < best_particle.current_loss:
                best_particle = particles[i]
        return best_particle

    def update_velocity(self, particle, inertia_weight, acc_c):
        # TODO: Look into clamping the velocity
        initial = (inertia_weight) * (particle.velocity)
        cognitive_component = (acc_c[0]) * (random()) * (particle.pbest - particle.position)
        social_component = (acc_c[1]) * (random()) * (particle.gbest - particle.position)

        particle.velocity = initial + cognitive_component + social_component

    def calc_new_position(self, particle):
        return particle.position + particle.velocity
