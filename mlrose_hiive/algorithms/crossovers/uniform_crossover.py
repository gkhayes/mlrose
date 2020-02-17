""" Crossover implementations for GA.
"""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.algorithms.crossovers._crossover_base import _CrossOverBase


class UniformCrossOver(_CrossOverBase):
    def __init__(self, opt_prob):
        super().__init__(opt_prob)

    def mate(self, p1, p2):
        n = np.random.choice(a=[False, True], size=self._length)
        child = np.array([p1[i] if n[i] else p2[i] for i in range(self._length)])
        return child
