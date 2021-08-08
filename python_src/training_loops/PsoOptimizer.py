from random import uniform, seed

from tensorflow import Variable
from tensorflow._api.v2 import math
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.losses import CategoricalCrossentropy

from training_loops.OptimizerHelper import get_trainable_weights, set_trainable_weights, calc_solution_fitness

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


class PsoEnv():
    def __init__(self, fitness_function=calc_solution_fitness, iterations=5, swarm_size=8, model=None, X=None, y=None, layers_to_optimize=(Conv2D, Dense)):
        self.fitness_function = fitness_function
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
        best_weights = self.convert_tenor_weights_to_tf_variable(swarm[0].gbest)

        return set_trainable_weights(self.model, best_weights, self.layers_to_optimize)

    def initialize_swarm(self, swarm_size, weights, model, loss_metric, X, y):
        particles = [None] * swarm_size
        starting_velocity = [[w * 0 for w in weight] for weight in weights]
        particles[0] = Particle(weights, self.fitness_function(weights, model, loss_metric, X, y), starting_velocity)
        for p in range(1, swarm_size):
            new_weights = [[w * uniform(0, 1) for w in weight] for weight in weights]
            initial_fitness = self.fitness_function(new_weights, model, loss_metric, X, y)
            particles[p] = Particle(position=new_weights, fitness=initial_fitness, velocity=starting_velocity)
        return particles

    def convert_tenor_weights_to_tf_variable(self, weights):
        for r in range(len(weights)):
            for c in range(len(weights[r])):
                weights[r][c] = Variable(weights[r][c])
        return weights

    def set_gbest(self, particles, best_particle):
        for particle in particles:
            particle.gbest = best_particle.position
            particle.gbest_fitness = best_particle.best_fitness

    def update_positions(self, particles, model, loss_metric, X, y):
        for particle in particles:
            particle.velocity = self.update_velocity(particle, INERTIA, [C1, C2])
            particle.position = self.calc_new_position(particle)
            particle.current_fitness = self.fitness_function(particle.position, model, loss_metric, X, y)
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
        initial = [[inertia_weight * v for v in velocity] for velocity in particle.velocity]
        cognitive_component = self.get_cognitive_component(particle, acc_c)
        social_component = self.get_social_component(particle, acc_c)

        return self.add_three_tensors(initial, cognitive_component, social_component)

    def get_cognitive_component(self, particle, acc_c):
        magic = (acc_c[0]) * (uniform(0, 1))
        cognitive_component = self.perform_tensor_operations(math.subtract, particle.pbest, particle.position)
        cognitive_component = [[c * magic for c in component] for component in cognitive_component]
        return cognitive_component

    def get_social_component(self, particle, acc_c):
        magic = (acc_c[1]) * (uniform(0, 1))
        social_component = self.perform_tensor_operations(math.subtract, particle.gbest, particle.position)
        social_component = [[c * magic for c in component] for component in social_component]
        return social_component

    def perform_tensor_operations(self, operation_function, tensor_1, tensor_2):
        new_tensor = []
        for i in range(len(tensor_1)):
            vars = []
            for n in range(len(tensor_1[i])):
                vars.append(operation_function(tensor_1[i][n], tensor_2[i][n]))
            new_tensor.append(
                vars
            )
        return new_tensor

    def add_three_tensors(self, tensor_1, tensor_2, tensor_3):
        new_tensor = []
        for i in range(len(tensor_1)):
            vars = []
            for n in range(len(tensor_1[i])):
                vars.append(tensor_1[i][n] + tensor_2[i][n] + tensor_3[i][n])
            new_tensor.append(
                vars
            )
        return new_tensor

    def create_empty_tensor_with_same_shape(self, tensor):
        new_tensor = []
        for i in range(len(tensor)):
            vars = []
            for x in tensor:
                vars.append(0 * x)
            new_tensor.append(
                vars
            )
        return new_tensor

    def calc_new_position(self, particle):
        return self.perform_tensor_operations(math.add, particle.position, particle.velocity)
