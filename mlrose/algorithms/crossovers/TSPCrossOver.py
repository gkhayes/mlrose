"""
TSP Crossover implementation for GA.
"""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose.algorithms.crossovers._CrossOverBase import _CrossOverBase


class TSPCrossOver(_CrossOverBase):

    def __init__(self, opt_prob):
        super().__init__(opt_prob)

    def mate(self, p1, p2):
        if self._length > 1:
            n = 1 + np.random.randint(self._length - 1)
            child = np.array([0] * self._length)
            child[:n] = p1[:n]

            unvisited = [node for node in p2 if node not in p1[:n]]
            child[n:] = unvisited
        elif np.random.randint(2) == 0:
            child = np.copy(p1)
        else:
            child = np.copy(p2)
        return child