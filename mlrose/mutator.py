import numpy as np
from abc import ABC, abstractmethod


class _MutatorBase(ABC):
    def __init__(self, opt_prob):
        self._opt_prob = opt_prob
        self._length = opt_prob.length

    @abstractmethod
    def mutate(self, child, mutation_probability):
        pass


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


class SwapMutator(_MutatorBase):

    def __init__(self, opt_prob):
        super().__init__(opt_prob)

    def mutate(self, child, mutation_probability):
        if np.random.rand() > mutation_probability:
            # do swap mutation
            m1 = np.random.randint(len(child))
            m2 = np.random.randint(len(child))
            tmp = child[m1]
            child[m1] = child[m2]
            child[m2] = tmp
        return child


class ChangeOneMutator(_MutatorBase):

    def __init__(self, opt_prob):
        super().__init__(opt_prob)
        self._max_val = opt_prob.max_val

    def mutate(self, child, mutation_probability):
        if np.random.rand() > mutation_probability:
            # do swap mutation
            m = np.random.randint(len(child))

            child[m] = np.random.randint(self._max_val + 1)
        return child
