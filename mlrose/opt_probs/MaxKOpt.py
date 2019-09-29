""" Classes for defining optimization problem objects."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose.algorithms.crossovers import UniformCrossOver
from mlrose.algorithms.mutators import ChangeOneMutator
from mlrose.fitness import MaxKColor
from mlrose.opt_probs.DiscreteOpt import DiscreteOpt

import networkx as nx


class MaxKOpt(DiscreteOpt):
    def __init__(self, edges=None, length=None, fitness_fn=None, maximize=False,
                 crossover=None, mutator=None):

        if (fitness_fn is None) and (edges is None):
            raise Exception("fitness_fn or edges must be specified.")

        if length is None:
            if fitness_fn is None:
                length = len(edges)
            else:
                length = len(fitness_fn.weights)

        self.length = length

        if fitness_fn is None:
            fitness_fn = MaxKColor(edges)

        self.max_val = length
        crossover = UniformCrossOver(self) if crossover is None else crossover
        mutator = ChangeOneMutator(self) if mutator is None else mutator
        super().__init__(length, fitness_fn, maximize, length, crossover, mutator)

        # set up initial state (everything painted one color)
        g = nx.Graph()
        g.add_edges_from(edges)
        fitness_fn.set_graph(g)

        # state = [len([*g.neighbors(n)]) for n in range(length)]
        # state = np.random.randint(self.length, size=self.length)
        # np.random.shuffle(state)
        state = [0] * length
        self.set_state(state)

    def can_stop(self):
        return int(self.get_fitness()) == 0
