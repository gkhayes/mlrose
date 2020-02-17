""" Classes for defining fitness functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np


class Knapsack:
    """Fitness function for Knapsack optimization problem. Given a set of n
    items, where item i has known weight :math:`w_{i}` and known value
    :math:`v_{i}`; and maximum knapsack capacity, :math:`W`, the Knapsack
    fitness function evaluates the fitness of a state vector
    :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]` as:

    .. math::
        Fitness(x) = \\sum_{i = 0}^{n-1}v_{i}x_{i}, \\text{ if}
        \\sum_{i = 0}^{n-1}w_{i}x_{i} \\leq W, \\text{ and 0, otherwise,}

    where :math:`x_{i}` denotes the number of copies of item i included in the
    knapsack.

    Parameters
    ----------
    weights: list
        List of weights for each of the n items.

    values: list
        List of values for each of the n items.

    max_weight_pct: float, default: 0.35
        Parameter used to set maximum capacity of knapsack (W) as a percentage
        of the total of the weights list
        (:math:`W =` max_weight_pct :math:`\\times` total_weight).

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose_hiive
        >>> import numpy as np
        >>> weights = [10, 5, 2, 8, 15]
        >>> values = [1, 2, 3, 4, 5]
        >>> max_weight_pct = 0.6
        >>> fitness = mlrose_hiive.Knapsack(weights, values, max_weight_pct)
        >>> state = np.array([1, 0, 2, 1, 0])
        >>> fitness.evaluate(state)
        11

    Note
    ----
    The Knapsack fitness function is suitable for use in discrete-state
    optimization problems *only*.
    """

    def __init__(self, weights, values, max_weight_pct=0.35, max_item_count=1, multiply_by_max_item_count=False):

        self.weights = weights
        self.values = values
        count_multiplier = max_item_count if multiply_by_max_item_count else 1.0
        self._w = np.ceil(np.sum(self.weights) * max_weight_pct * count_multiplier)
        self.prob_type = 'discrete'

        if len(self.weights) != len(self.values):
            raise Exception("""The weights array and values array must be"""
                            + """ the same size.""")

        if min(self.weights) <= 0:
            raise Exception("""All weights must be greater than 0.""")

        if min(self.values) <= 0:
            raise Exception("""All values must be greater than 0.""")

        if max_item_count <= 0:
            raise Exception("""max_item_count must be greater than 0.""")

        if max_weight_pct <= 0:
            raise Exception("""max_weight_pct must be greater than 0.""")

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation. Must be the same length as the weights
            and values arrays.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        if len(state) != len(self.weights):
            raise Exception("""The state array must be the same size as the"""
                            + """ weight and values arrays.""")

        # Calculate total weight and value of knapsack
        total_weight = np.sum(state*self.weights)
        total_value = np.sum(state*self.values)

        # Allow for weight constraint
        if total_weight <= self._w:
            fitness = total_value
        else:
            fitness = 0

        return fitness

    def get_prob_type(self):
        """ Return the problem type.

        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp'
            or 'either'.
        """
        return self.prob_type
