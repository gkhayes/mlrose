""" GA Mutators."""

# Author: Genevieve Hayes
# Modified: Andrew Rollings
# License: BSD 3 clause
import numpy as np

from mlrose_hiive.algorithms.mutators._mutator_base import _MutatorBase


class DiscreteMutator(_MutatorBase):

    def __init__(self, opt_prob):
        super().__init__(opt_prob)
        self._max_val = opt_prob.max_val

    def mutate(self, child, mutation_probability):
        rand = np.random.uniform(size=self._length)
        mutate = np.where(rand < mutation_probability)[0]

        if self._max_val == 2:
            for i in mutate:
                child[i] = np.abs(child[i] - 1)

        else:
            for i in mutate:
                vals = list(np.arange(self._max_val))
                vals.remove(child[i])
                child[i] = vals[np.random.randint(0, self._max_val-1)]
        return child
