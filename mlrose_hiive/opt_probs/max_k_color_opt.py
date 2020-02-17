""" Classes for defining optimization problem objects."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.algorithms.crossovers import UniformCrossOver
from mlrose_hiive.algorithms.mutators import ChangeOneMutator
from mlrose_hiive.fitness import MaxKColor
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt

import networkx as nx


class MaxKColorOpt(DiscreteOpt):
    def __init__(self, edges=None, length=None, fitness_fn=None, maximize=False,
                 max_colors=None, crossover=None, mutator=None):

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

        # set up initial state (everything painted one color)
        g = nx.Graph()
        g.add_edges_from(edges)
        fitness_fn.set_graph(g)

        # if none is provided, make a reasonable starting guess.
        # the max val is going to be the one plus the maximum number of neighbors of any one node.
        if max_colors is None:
            max_colors = 1 + max([len([*g.neighbors(n)]) for n in range(length)])
        self.max_val = max_colors

        crossover = UniformCrossOver(self) if crossover is None else crossover
        mutator = ChangeOneMutator(self) if mutator is None else mutator
        super().__init__(length, fitness_fn, maximize, max_colors, crossover, mutator)

        # state = [len([*g.neighbors(n)]) for n in range(length)]
        state = np.random.randint(max_colors, size=self.length)
        np.random.shuffle(state)
        # state = [0] * length
        self.set_state(state)

    def can_stop(self):
        return int(self.get_fitness()) == 0
