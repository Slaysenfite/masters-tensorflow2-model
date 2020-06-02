from random import random, randint, seed

seed(1)
rand_seed = randint(0, 10)
seed(rand_seed)


class Particle:
    def __init__(self, position, velocity):
        self.velocity = velocity
        self.position = position
        self.gbest = position
        self.pbest = position


#TODO: Implement this method
def calculate_loss(weights):
    pass


class Position:
    def __init__(self, weights):
        self.weights = weights
        self.loss = calculate_loss(weights)


def update_velocity(particle, inertia_weight, acc_c):
    # TODO: Look into clamping the velocity
    return (inertia_weight)(particle.velocity) \
           + (acc_c[0])(random())(particle.pbest - particle.position) \
           + (acc_c[1])(random())(particle.gbest - particle.position)


def update_position(particle, inertia_weight, acc_c):
    return particle.position + update_position(particle, inertia_weight, acc_c)

# def calculate_loss():
