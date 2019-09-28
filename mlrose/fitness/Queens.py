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

        >>> import mlrose
        >>> import numpy as np
        >>> fitness = mlrose.Queens()
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

        fitness = 0

        for i in range(len(state) - 1):
            for j in range(i + 1, len(state)):
                # Check for horizontal attacks
                if state[j] == state[i]:
                    fitness += 1

                # Check for diagonal-up attacks
                elif state[j] == state[i] + (j - i):
                    fitness += 1

                # Check for diagonal-down attacks
                elif state[j] == state[i] - (j - i):
                    fitness += 1

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
