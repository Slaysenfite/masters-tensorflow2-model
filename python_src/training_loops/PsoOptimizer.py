from random import uniform, seed

import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import CategoricalCrossentropy

seed(1)

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
        self.gbest_loss = None


class PsoEnv():
    def __init__(self, iterations, swarm_size, model, X, y):
        self.iterations = iterations
        self.swarm_size = swarm_size
        self.model = model
        self.X = X
        self.y = y

    def get_pso_model(self):
        iteration = 0;
        loss_metric = CategoricalCrossentropy(from_logits=True)
        weights = self.get_trainable_weights(self.model)

        swarm = self.initialize_swarm(self.swarm_size, weights, self.model, loss_metric, self.X, self.y)

        best_particle = self.find_best_particle(swarm)
        self.set_gbest(swarm, best_particle)

        # TODO: Figure out why best loss is not transfering from batch to batch
        # Unless it is transfering but the loss changes because of the inputs
        while iteration < self.iterations:
            print(' PSO training for iteration {}'.format(iteration + 1))

            self.update_positions(swarm, self.model, loss_metric, self.X, self.y)

            self.update_gbest(swarm)

            print(' Best loss of {} for iteration {}'.format(swarm[0].gbest_loss, iteration + 1))
            iteration += 1
        best_weights = swarm[0].gbest

        swarm = None

        return self.set_trainable_weights(self.model, best_weights)

    def initialize_swarm(self, swarm_size, weights, model, loss_metric, X, y):
        particles = [None] * swarm_size
        particles[0] = Particle(weights, self.calc_position_loss(weights, model, loss_metric, X, y))
        for p in range(1, swarm_size):
            new_weights = [w * uniform(-1, 1) for w in weights]
            initial_loss = self.calc_position_loss(new_weights, model, loss_metric, X, y)
            particles[p] = Particle(np.array(new_weights), initial_loss)
        return particles

    def calc_position_loss(self, weights, model, loss_metric, X, y):
        self.set_trainable_weights(model, weights)
        ŷ = model(X, training=True)
        return loss_metric(y, ŷ).numpy()

    def set_gbest(self, particles, best_particle):
        for particle in particles:
            particle.gbest = best_particle.position
            particle.gbest_loss = best_particle.best_loss

    def update_positions(self, particles, model, loss_metric, X, y):
        for particle in particles:
            particle.velocity = self.update_velocity(particle, INERTIA, [C1, C2])
            particle.position = self.calc_new_position(particle)
            particle.current_loss = self.calc_position_loss(particle.position, model, loss_metric, X, y)
            if particle.current_loss < particle.best_loss:
                particle.pbest = particle.position
                particle.best_loss = particle.current_loss

    def update_gbest(self, swarm):
        best_particle = swarm[0]
        initial_best_loss = best_particle.gbest_loss
        for i in range(1, len(swarm)):
            if swarm[i].best_loss < best_particle.gbest_loss:
                best_particle = swarm[i]
        if best_particle.best_loss < initial_best_loss:
            self.set_gbest(swarm, best_particle)


    def get_trainable_weights(self, model):
        weights = []
        for layer in model.layers:
            if (layer.trainable != True or len(layer.trainable_weights) == 0):
                pass
            if isinstance(layer, (Dense)):
                weights.append(layer.get_weights())
        return np.array(weights)

    def set_trainable_weights(self, model, weights):
        i = 0
        for layer in model.layers:
            if (layer.trainable != True or len(layer.weights) == 0):
                pass
            if isinstance(layer, (Dense)):
                layer.set_weights(weights[i])
                i += 1
        return model

    def find_best_particle(self, particles):
        best_particle = particles[0]
        for i in range(1, len(particles)):
            if particles[i].current_loss < best_particle.current_loss:
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
