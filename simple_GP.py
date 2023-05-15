import random
from typing import *
import numpy as np

import matplotlib.pyplot as plt


# Define the target function
def target_function(x):
    return x ** 2 + x + 1


# Define primitive functions
def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def mul(x, y):
    return x * y


# TODO: implement a simple GP algorithm here to find an expression with one input (independent variable x), whose output equals the value of the target function

class element:
    def __init__(self, l_child, r_child, value: Callable | float):
        self.value: Callable | float | None = value  # None represent x
        # self.parent = parent
        self.l_child: element = l_child
        self.r_child: element = r_child

    def calculate(self, x):
        if not callable(self.value): return self.value if self.value is not None else x
        return self.value(self.l_child.calculate(x), self.r_child.calculate(x))

    def simplify(self):
        if not(callable(self.l_child) and callable(self.r_child)):
            self.l_child = self.l_child.simplify()
            self.r_child = self.r_child.simplify()
        elif callable(self.l_child) and not callable(self.r_child):
            pass
        elif not callable(self.l_child) and callable(self.r_child):
            pass
        else:
            if self.l_child.value == 0 or self.r_child.value == 0:
                return 0
            elif self.l_child.value == 1:
                return self.r_child
            elif self.r_child.value == 1:
                return self.l_child
            else:
                return self


    def __str__(self):
        if not callable(self.value): return str(self.value) if self.value is not None else "x"
        return f"({self.l_child.__str__()} {self.value.__name__} {self.r_child.__str__()})"


funcs = [add, sub, mul]

def init_tree(dmax=5):
    def init_element(depth):
        if depth >= dmax - 1: return element(None, None, random.choice([random.randrange(0, 2), None]))
        return element(init_element(depth + 1), init_element(depth + 1), random.choice(funcs))

    root: element = init_element(0)
    return root


def fitness(ele: element, points):
    error = 0
    for x in points:
        error += abs(target_function(x) - ele.calculate(x))
    return error

def fitness_with_points(ele: element):
    points = [i/10. for i in range(-10, 10)]
    return fitness(ele, points)

def init_population(pop_size=100):
    population = []

    for i in range(pop_size):
        population.append(init_tree(5))

    return population


def crossover(element1: element, element2: element):
    off1 = element(element1.l_child, element2.r_child, element1.value)
    off2 = element(element2.l_child, element1.r_child, element2.value)
    return off1, off2


def mutation(ele: element, mutate_prob):
    if random.random() < mutate_prob:
        return init_tree(4)
    if not callable(ele.value):
        ele.l_child = mutation(ele.l_child, mutate_prob)
        ele.r_child = mutation(ele.r_child, mutate_prob)
        return ele
    return ele

    # curr_node = ele
    # while callable(curr_node.l_child.value) and callable(curr_node.r_child.value):
    #     curr_node = curr_node.l_child if random.uniform(0, 1) < 0.5 else curr_node.r_child
    #
    # # now behind root node
    # new_subnode = init_tree(random.randint(1, 3))
    #
    # if callable(curr_node.l_child.value):
    #     curr_node.l_child = new_subnode
    # else:
    #     curr_node.r_child = new_subnode

def run_genetic_programming(population_size, crossover_probability, mutation_probability, number_of_generations,
                            max_depth, elitism_size):
    best_fitness_values = []

    # Initialize the population
    population = init_population(population_size)
    points = [x / 10. for x in range(-10, 10)]

    for generation in range(number_of_generations):
        print(f"Generation {generation + 1}/{number_of_generations}")

        # Evaluate the population
        fitness_values = [fitness(ind, points) for ind in population]
        # Select the best individuals for elitism
        elite_individuals = sorted(population, key=fitness_with_points)[:elitism_size]

        # Select parents
        parents = []
        for _ in range((population_size - elitism_size) // 2):
            p1 = min(random.sample(population, 10), key=fitness_with_points)  # Increase selection pressure
            p2 = min(random.sample(population, 10), key=fitness_with_points)  # Increase selection pressure
            parents.append((p1, p2))

        # Create offspring
        offspring = []
        for parent1, parent2 in parents:
            if random.random() < crossover_probability:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            offspring.append(mutation(child1, mutation_probability))
            offspring.append(mutation(child2, mutation_probability))

        population = elite_individuals + offspring

        best_fitness = min(fitness_values)
        best_fitness_values.append(best_fitness)

    plt.plot(best_fitness_values)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Fitness Curve")
    plt.show()

    best_individual = min(population, key=lambda ind: fitness(ind, points))
    print("Best individual:", best_individual)
    print("Fitness:", fitness(best_individual, points))



if __name__ == '__main__2':
    # population = init_population(100)
    ele = init_tree(4)
    print(ele)
    # for iter in range(10000):

if __name__ == '__main__':
    population_size = 100
    crossover_probability = 0.8
    mutation_probability = 0.3
    number_of_generations = 50
    max_depth = 6
    elitism_size = 5

    # Run the genetic programming algorithm with the given hyperparameters
    run_genetic_programming(population_size, crossover_probability, mutation_probability, number_of_generations,
                            max_depth, elitism_size)