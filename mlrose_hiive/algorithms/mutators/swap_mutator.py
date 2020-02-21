""" GA Mutators."""

# Author: Genevieve Hayes
# Modified: Andrew Rollings
# License: BSD 3 clause
import numpy as np

from mlrose_hiive.algorithms.mutators._mutator_base import _MutatorBase


class SwapMutator(_MutatorBase):

    def __init__(self, opt_prob):
        super().__init__(opt_prob)

    def mutate(self, child, mutation_probability):
        if np.random.rand() < mutation_probability:
            # do swap mutation
            m1 = np.random.randint(len(child))
            m2 = np.random.randint(len(child))
            tmp = child[m1]
            child[m1] = child[m2]
            child[m2] = tmp
        return child
