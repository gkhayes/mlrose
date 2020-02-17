""" Classes for defining optimization problem objects."""

# Author: Genevieve Hayes
# License: BSD 3 clause

import numpy as np

import mlrose_hiive


class KnapsackGenerator:
    @staticmethod
    def generate(seed, number_of_items_types=10,
                 max_item_count=5, max_weight_per_item=25,
                 max_value_per_item=10, max_weight_pct=0.6,
                 multiply_by_max_item_count=True):
        np.random.seed(seed)
        weights = 1 + np.random.randint(max_weight_per_item, size=number_of_items_types)
        values = 1 + np.random.randint(max_value_per_item, size=number_of_items_types)
        problem = mlrose_hiive.KnapsackOpt(length=number_of_items_types,
                                           maximize=True, max_val=max_item_count,
                                           weights=weights, values=values,
                                           max_weight_pct=max_weight_pct,
                                           multiply_by_max_item_count=multiply_by_max_item_count)
        return problem
