""" GA Mutators."""

# Author: Genevieve Hayes
# Modified: Andrew Rollings
# License: BSD 3 clause
from abc import ABC, abstractmethod


class _MutatorBase(ABC):
    def __init__(self, opt_prob):
        self._opt_prob = opt_prob
        self._length = opt_prob.length

    @abstractmethod
    def mutate(self, child, mutation_probability):
        pass
