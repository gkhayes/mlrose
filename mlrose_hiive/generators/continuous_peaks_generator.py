""" Classes for defining optimization problem objects."""

# Author: Andrew Rollings
# License: BSD 3 clause

import numpy as np

from mlrose_hiive import DiscreteOpt, ContinuousPeaks


class ContinuousPeaksGenerator:
    @staticmethod
    def generate(seed, size=20, t_pct=0.1):
        np.random.seed(seed)
        fitness = ContinuousPeaks(t_pct=t_pct)
        problem = DiscreteOpt(length=size, fitness_fn=fitness)
        return problem
