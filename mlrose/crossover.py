import numpy as np
from abc import ABC, abstractmethod


class _CrossOverBase(ABC):
    def __init__(self, opt_prob):
        self._opt_prob = opt_prob
        self._length = opt_prob.length

    @abstractmethod
    def mate(self, p1, p2):
        pass


class UniformCrossOver(_CrossOverBase):
    def __init__(self, opt_prob):
        super().__init__(opt_prob)

    def mate(self, p1, p2):
        bs = np.random.choice(a=[False, True], size=self._length)
        child = np.array([0] * self._length)
        for i in range(len(bs)):
            child[i] = p1[i] if bs[i] else p2[i]
        return child


class TSPCrossOver(_CrossOverBase):

    def __init__(self, opt_prob):
        super().__init__(opt_prob)

    def mate(self, p1, p2):
        if self._length > 1:
            _n = np.random.randint(self._length - 1)
            child = np.array([0] * self._length)
            child[0:_n + 1] = p1[0:_n + 1]

            unvisited = [node for node in p2 if node not in p1[0:_n + 1]]
            child[_n + 1:] = unvisited
        elif np.random.randint(2) == 0:
            child = np.copy(p1)
        else:
            child = np.copy(p2)
        return child