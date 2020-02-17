""" Classes for defining optimization problem objects."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

from mlrose_hiive.algorithms.crossovers import UniformCrossOver
from mlrose_hiive.algorithms.mutators import ChangeOneMutator
from mlrose_hiive.fitness.knapsack import Knapsack
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt


class KnapsackOpt(DiscreteOpt):
    def __init__(self, length=None, fitness_fn=None, maximize=True, max_val=2,
                 weights=None, values=None, max_weight_pct=0.35,
                 crossover=None, mutator=None,
                 multiply_by_max_item_count=False):

        if (fitness_fn is None) and (weights is None and values is None):
            raise Exception("""fitness_fn or both weights and"""
                            + """ values must be specified.""")

        if length is None:
            if weights is not None:
                length = len(weights)
            elif values is not None:
                length = len(values)
            elif fitness_fn is not None:
                length = len(fitness_fn.weights)

        self.length = length

        if fitness_fn is None:
            fitness_fn = Knapsack(weights=weights, values=values,
                                  max_weight_pct=max_weight_pct,
                                  max_item_count=max_val,
                                  multiply_by_max_item_count=multiply_by_max_item_count)

        self.max_val = max_val
        crossover = UniformCrossOver(self) if crossover is None else crossover
        mutator = ChangeOneMutator(self) if mutator is None else mutator
        super().__init__(length, fitness_fn, maximize, max_val, crossover, mutator)
