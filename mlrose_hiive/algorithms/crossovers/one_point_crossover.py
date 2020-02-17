""" Crossover implementations for GA.
"""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.algorithms.crossovers._crossover_base import _CrossOverBase


class OnePointCrossOver(_CrossOverBase):
    def __init__(self, opt_prob):
        super().__init__(opt_prob)

    def mate(self, p1, p2):
        n = 1 + np.random.randint(self._length-1)
        child = np.array([*p1[:n], *p2[n:]])
        return child
