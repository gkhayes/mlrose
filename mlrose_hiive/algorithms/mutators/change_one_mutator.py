""" GA Mutators."""

# Author: Genevieve Hayes
# Modified: Andrew Rollings
# License: BSD 3 clause
import numpy as np

from mlrose_hiive.algorithms.mutators._mutator_base import _MutatorBase


class ChangeOneMutator(_MutatorBase):

    def __init__(self, opt_prob):
        super().__init__(opt_prob)
        self._max_val = opt_prob.max_val

    def mutate(self, child, mutation_probability):
        if np.random.rand() < mutation_probability:
            # do change one mutation
            m = np.random.randint(len(child))

            child[m] = np.random.randint(self._max_val)
        return child
