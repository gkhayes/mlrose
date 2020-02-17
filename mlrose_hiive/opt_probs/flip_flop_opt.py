""" Classes for defining optimization problem objects."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.algorithms.crossovers import OnePointCrossOver
from mlrose_hiive.algorithms.mutators import ChangeOneMutator
from mlrose_hiive.fitness import FlipFlop
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt


class FlipFlopOpt(DiscreteOpt):
    def __init__(self, length=None, fitness_fn=None, maximize=True,
                 crossover=None, mutator=None):

        if (fitness_fn is None) and (length is None):
            raise Exception("fitness_fn or length must be specified.")

        if length is None:
            length = len(fitness_fn.weights)

        self.length = length

        if fitness_fn is None:
            fitness_fn = FlipFlop()

        self.max_val = 2
        crossover = OnePointCrossOver(self) if crossover is None else crossover
        mutator = ChangeOneMutator(self) if mutator is None else mutator
        super().__init__(length, fitness_fn, maximize, 2, crossover, mutator)

        state = np.random.randint(2, size=self.length)
        self.set_state(state)

    def evaluate_population_fitness(self):
        # Calculate fitness
        pop_fitness = self.fitness_fn.evaluate_many(self.population)
        self.pop_fitness = pop_fitness

    def random_pop(self, pop_size):
        """Create a population of random state vectors.

        Parameters
        ----------
        pop_size: int
            Size of population to be created.
        """
        if pop_size <= 0:
            raise Exception("""pop_size must be a positive integer.""")
        elif not isinstance(pop_size, int):
            if pop_size.is_integer():
                pop_size = int(pop_size)
            else:
                raise Exception("""pop_size must be a positive integer.""")

        """
        population = []


        for _ in range(pop_size):
            state = self.random()
            population.append(state)

        self.population = np.array(population)
        """
        population = np.random.rand(pop_size, self.length)
        population[population < 0.5] = 0
        population[population >= 0.5] = 1
        self.population = population
        # np.round(population, out=population).astype(int)

        self.evaluate_population_fitness()

    def can_stop(self):
        return int(self.get_fitness()) == int(self.length - 1)
