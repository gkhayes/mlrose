""" Classes for defining optimization problem objects."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose.algorithms.crossovers import OnePointCrossOver
from mlrose.algorithms.mutators import ChangeOneMutator
from mlrose.fitness import FlipFlop
from mlrose.opt_probs.DiscreteOpt import DiscreteOpt


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

    def can_stop(self):
        return int(self.get_fitness()) == int(self.length - 1)
