""" Crossover implementations for GA.
"""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause


from abc import ABC, abstractmethod


class _CrossOverBase(ABC):
    def __init__(self, opt_prob):
        self._opt_prob = opt_prob
        self._length = opt_prob.length

    @abstractmethod
    def mate(self, p1, p2):
        pass
