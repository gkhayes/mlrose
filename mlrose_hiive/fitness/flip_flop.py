""" Classes for defining fitness functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np


class FlipFlop:
    """Fitness function for Flip Flop optimization problem. Evaluates the
    fitness of a state vector :math:`x` as the total number of pairs of
    consecutive elements of :math:`x`, (:math:`x_{i}` and :math:`x_{i+1}`)
    where :math:`x_{i} \\neq x_{i+1}`.

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose_hiive
        >>> import numpy as np
        >>> fitness = mlrose_hiive.FlipFlop()
        >>> state = np.array([0, 1, 0, 1, 1, 1, 1])
        >>> fitness.evaluate(state)
        3

    Note
    ----
    The Flip Flop fitness function is suitable for use in discrete-state
    optimization problems *only*.
    """

    def __init__(self):

        self.prob_type = 'discrete'

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

        fitness = sum([state[i] != state[i-1] for i in range(1, len(state))])


        # may not be faster
        """
        runs = np.zeros(state.size-1, dtype=int)
        np.not_equal(state[:-1], state[1:], out=runs)
        fitness = np.sum(runs)
        """

        return fitness

    def evaluate_many(self, states):
        """Evaluate the fitness of a an ndarray of states.

        Parameters
        ----------
        states: ndarray
            States array for evaluation.

        Returns
        -------
        fitness: ndarray
            Population fitness values.
        """

        runs = np.zeros((states.shape[0], states.shape[1]-1), dtype=int)
        np.not_equal(states[:, :-1], states[:, 1:], out=runs)
        fitness = np.sum(runs, axis=1)

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
