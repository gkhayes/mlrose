""" Classes for defining fitness functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np


class ContinuousPeaks:
    """Fitness function for Continuous Peaks optimization problem. Evaluates
    the fitness of an n-dimensional state vector :math:`x`, given parameter T,
    as:

    .. math::
        Fitness(x, T) = \\max(max\\_run(0, x), max\\_run(1, x)) + R(x, T)

    where:

    * :math:`max\\_run(b, x)` is the length of the maximum run of b's
      in :math:`x`;
    * :math:`R(x, T) = n`, if (:math:`max\\_run(0, x) > T` and
      :math:`max\\_run(1, x) > T`); and
    * :math:`R(x, T) = 0`, otherwise.

    Parameters
    ----------
    t_pct: float, default: 0.1
        Threshold parameter (T) for Continuous Peaks fitness function,
        expressed as a percentage of the state space dimension, n (i.e.
        :math:`T = t_{pct} \\times n`).

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose_hiive
        >>> import numpy as np
        >>> fitness = mlrose_hiive.ContinuousPeaks(t_pct=0.15)
        >>> state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1])
        >>> fitness.evaluate(state)
        17

    Note
    ----
    The Continuous Peaks fitness function is suitable for use in bit-string
    (discrete-state with :code:`max_val = 2`) optimization problems *only*.
    """

    def __init__(self, t_pct=0.1):

        self.t_pct = t_pct
        self.prob_type = 'discrete'

        if (self.t_pct < 0) or (self.t_pct > 1):
            raise Exception("""t_pct must be between 0 and 1.""")

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
        _n = len(state)
        _t = np.ceil(self.t_pct*_n)

        # Calculate length of maximum runs of 0's and 1's
        max_0 = self.max_run(0, state)
        max_1 = self.max_run(1, state)

        # Calculate R(X, T)
        if (max_0 > _t and max_1 > _t):
            _r = _n
        else:
            _r = 0

        # Evaluate function
        fitness = max(max_0, max_1) + _r

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

    @staticmethod
    def max_run(_b, _x):
        """Determine the length of the maximum run of b's in vector x.

        Parameters
        ----------
        b: int
            Integer for counting.

        x: array
            Vector of integers.

        Returns
        -------
        max: int
            Length of maximum run of b's.
        """
        # Initialize counter
        _max = 0
        run = 0

        # Iterate through values in vector
        for i in _x:
            if i == _b:
                run += 1
            else:
                if run > _max:
                    _max = run

                run = 0

        if (_x[-1] == _b) and (run > _max):
            _max = run

        return _max
