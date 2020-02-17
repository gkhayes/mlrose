"""
TSP Crossover implementation for GA.
"""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.algorithms.crossovers._crossover_base import _CrossOverBase


class TSPCrossOver(_CrossOverBase):

    def __init__(self, opt_prob):
        super().__init__(opt_prob)

    def mate(self, p1, p2):
        return self._mate_fill(p1, p2)
        """
        if np.random.randint(2) == 0:
            return self._mate_fill(p1, p2)
        else:
            return self._mate_traverse(p1, p2)
        """

    def _mate_fill(self, p1, p2):
        if self._length > 1:
            n = 1 + np.random.randint(self._length - 1)
            child = np.array([0] * self._length)
            child[:n] = p1[:n]

            unvisited = [node for node in p2 if node not in p1[:n]]
            child[n:] = unvisited
        elif np.random.randint(2) == 0:
            child = np.copy(p1)
        else:
            child = np.copy(p2)
        return child

    def _mate_traverse(self, parent_1, parent_2):
        if self._length > 1:
            next_a = np.append(parent_1[1:], parent_1[-1])
            next_b = np.append(parent_2[1:], parent_2[-1])

            visited = [False] * self._length
            child = np.array([0] * self._length)

            v = np.random.randint(len(parent_1))
            child[0] = v
            visited[v] = True
            for i in range(1, len(child)):
                cur = child[i-1]
                na = next_a[cur]
                nb = next_b[cur]
                va = visited[na]
                vb = visited[nb]
                if va and not vb:
                    nx = nb
                elif not va and vb:
                    nx = na
                elif not va and not vb:
                    fa = self._opt_prob.fitness_fn.calculate_fitness([cur, na])
                    fb = self._opt_prob.fitness_fn.calculate_fitness([cur, nb])
                    nx = nb if fa > fb else na  # opposite because they're distance and smaller is better
                else:
                    while True:
                        nx = np.random.randint(len(parent_1))
                        if not visited[nx]:
                            break
                child[i] = nx
                visited[nx] = True
        elif np.random.randint(2) == 0:
            child = np.copy(parent_1, copy=True)
        else:
            child = np.copy(parent_2, copy=True)
        return child
