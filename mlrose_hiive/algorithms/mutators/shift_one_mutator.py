""" GA Mutators."""

# Author: Genevieve Hayes
# Modified: Andrew Rollings
# License: BSD 3 clause
import numpy as np

from mlrose_hiive.algorithms.mutators._mutator_base import _MutatorBase


class ShiftOneMutator(_MutatorBase):

    def __init__(self, opt_prob):
        super().__init__(opt_prob)
        self._max_val = opt_prob.max_val

    def mutate(self, child, mutation_probability):
        if np.random.rand() < mutation_probability:
            # do shift one mutation
            m = np.random.randint(len(child))
            # bump value up or down
            new_val = child[m] + (1 if np.random.randint(2) == 0 else -1)
            # wrap around if greater than or less than max_val
            new_val = (self._max_val + new_val) % self._max_val
            child[m] = new_val
        return child
