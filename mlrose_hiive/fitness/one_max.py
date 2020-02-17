""" Classes for defining fitness functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np


class OneMax:
    """Fitness function for One Max optimization problem. Evaluates the
    fitness of an n-dimensional state vector
    :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]` as:

    .. math::
        Fitness(x) = \\sum_{i = 0}^{n-1}x_{i}

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose_hiive
        >>> import numpy as np
        >>> fitness = mlrose_hiive.OneMax()
        >>> state = np.array([0, 1, 0, 1, 1, 1, 1])
        >>> fitness.evaluate(state)
        5

    Note
    -----
    The One Max fitness function is suitable for use in either discrete or
    continuous-state optimization problems.
    """

    def __init__(self):

        self.prob_type = 'either'

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        fitness = np.sum(state)
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
