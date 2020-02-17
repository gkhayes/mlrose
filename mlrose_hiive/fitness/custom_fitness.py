""" Classes for defining fitness functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np


class CustomFitness:
    """Class for generating your own fitness function.

    Parameters
    ----------
    fitness_fn: callable
        Function for calculating fitness of a state with the signature
        :code:`fitness_fn(state, **kwargs)`.

    problem_type: string, default: 'either'
        Specifies problem type as 'discrete', 'continuous', 'tsp' or 'either'
        (denoting either discrete or continuous).

    kwargs: additional arguments
        Additional parameters to be passed to the fitness function.

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose_hiive
        >>> import numpy as np
        >>> def cust_fn(state, c): return c*np.sum(state)
        >>> kwargs = {'c': 10}
        >>> fitness = mlrose_hiive.CustomFitness(cust_fn, **kwargs)
        >>> state = np.array([1, 2, 3, 4, 5])
        >>> fitness.evaluate(state)
        150
    """

    def __init__(self, fitness_fn, problem_type='either', **kwargs):

        if problem_type not in ['discrete', 'continuous', 'tsp', 'either']:
            raise Exception("""problem_type does not exist.""")
        self.fitness_fn = fitness_fn
        self.problem_type = problem_type
        self.kwargs = kwargs

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

        fitness = self.fitness_fn(state, **self.kwargs)
        return fitness

    def get_prob_type(self):
        """ Return the problem type.

        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp'
            or 'either'.
        """
        return self.problem_type
