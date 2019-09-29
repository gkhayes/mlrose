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


        """
        lc = [[{
            'column':(j),
            'check contents of':i,
            'distance':(j-i),
            f'contents of {i}': state[i],
            f'contents of {j}': state[j],
            'c1':(state[i] + (j-i)) == state[j],
            'c2':(state[i] - (j-i)) == state[j]} for i in range(j)] for j in range(1, len(state))]
        
        rc = [[{
            'column':(j),
            'check contents of':i,
            'distance':(i-j),
            f'contents of {i}': state[i],
            f'contents of {j}': state[j],
            'c1':(state[i] + (i-j)) == state[j],
            'c2':(state[i] - (i-j)) == state[j]} for i in range(j+1,len(state))] for j in range(len(state)-1)]
        """
        f_h = sum([list(state).count(state[i])-1 for i in range(len(state))])

        f_ld = sum([sum([(int((state[i] + (j-i)) == state[j]) +
                          int((state[i] - (j-i)) == state[j]))
                         for i in range(j)])
                    for j in range(1, len(state))])

        f_rd = sum([sum([int((state[i] + (i-j)) == state[j]) +
                         int((state[i] - (i-j)) == state[j])
                         for i in range(j+1, len(state))])
                    for j in range(len(state)-1)])

        fitness = f_h + f_ld + f_rd
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
