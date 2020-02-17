""" Classes for defining fitness functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np


class Queens:
    """Fitness function for N-Queens optimization problem. Evaluates the
    fitness of an n-dimensional state vector
    :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]`, where :math:`x_{i}`
    represents the row position (between 0 and n-1, inclusive) of the 'queen'
    in column i, as the number of pairs of attacking queens.

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose_hiive
        >>> import numpy as np
        >>> fitness = mlrose_hiive.Queens()
        >>> state = np.array([1, 4, 1, 3, 5, 5, 2, 7])
        >>> fitness.evaluate(state)
        6

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.

    Note
    ----
    The Queens fitness function is suitable for use in discrete-state
    optimization problems *only*.
    """

    def __init__(self):

        self.prob_type = 'discrete'

    @staticmethod
    def shift(a, num, fill_value=np.nan):
        result = np.empty(a.shape)
        if num > 0:
            result[:num] = fill_value
            result[num:] = a[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = a[-num:]
        else:
            result[:] = a
        return result

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

        # check for horizontal matches.
        f_h = (np.unique(state, return_counts=True)[1]-1).sum()

        # check for diagonal matches.
        # look at the state_shifts to figure out how this works. (I'm quite pleased with it)
        ls = state.size
        # rows 0-3:   checking up left.
        # rows 4-7:   checking down right.
        # rows 8-11:  checking up right
        # rows 12-15: checking down left
        state_shifts = np.array([self.shift(state, i)+i for i in np.arange(1-ls, ls) if i != 0] +
                                [self.shift(state, -i)+i for i in np.arange(1-ls, ls) if i != 0])
        # state_shifts[(state_shifts < 0)] = np.NaN
        # state_shifts[(state_shifts >= ls)] = np.NaN

        f_d = np.sum(state_shifts == state) // 2  # each diagonal piece is counted twice
        fitness = f_h + f_d
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
