""" Classes for defining optimization problem objects."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.algorithms.crossovers import UniformCrossOver
from mlrose_hiive.algorithms.mutators import ChangeOneMutator
from mlrose_hiive.fitness.queens import Queens
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt


class QueensOpt(DiscreteOpt):
    def __init__(self, length=None, fitness_fn=None, maximize=False,
                 crossover=None, mutator=None):

        if (fitness_fn is None) and (length is None):
            raise Exception("fitness_fn or length must be specified.")

        if length is None:
            length = len(fitness_fn.weights)

        self.length = length

        if fitness_fn is None:
            fitness_fn = Queens()

        self.max_val = length
        crossover = UniformCrossOver(self) if crossover is None else crossover
        mutator = ChangeOneMutator(self) if mutator is None else mutator
        super().__init__(length, fitness_fn, maximize, length, crossover, mutator)

        state = np.random.randint(self.length, size=self.length)
        np.random.shuffle(state)
        self.set_state(state)

    def can_stop(self):
        return int(self.get_fitness()) == 0
