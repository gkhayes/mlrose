""" Classes for defining fitness functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.fitness._discrete_peaks_base import _DiscretePeaksBase


class FourPeaks(_DiscretePeaksBase):
    """Fitness function for Four Peaks optimization problem. Evaluates the
    fitness of an n-dimensional state vector :math:`x`, given parameter T, as:

    .. math::
        Fitness(x, T) = \\max(tail(0, x), head(1, x)) + R(x, T)

    where:

    * :math:`tail(b, x)` is the number of trailing b's in :math:`x`;
    * :math:`head(b, x)` is the number of leading b's in :math:`x`;
    * :math:`R(x, T) = n`, if :math:`tail(0, x) > T` and
      :math:`head(1, x) > T`; and
    * :math:`R(x, T) = 0`, otherwise.

    Parameters
    ----------
    t_pct: float, default: 0.1
        Threshold parameter (T) for Four Peaks fitness function, expressed as
        a percentage of the state space dimension, n (i.e.
        :math:`T = t_{pct} \\times n`).

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose_hiive
        >>> import numpy as np
        >>> fitness = mlrose_hiive.FourPeaks(t_pct=0.15)
        >>> state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        >>> fitness.evaluate(state)
        16

    References
    ----------
    De Bonet, J., C. Isbell, and P. Viola (1997). MIMIC: Finding Optima by
    Estimating Probability Densities. In *Advances in Neural Information
    Processing Systems* (NIPS) 9, pp. 424–430.

    Note
    ----
    The Four Peaks fitness function is suitable for use in bit-string
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
        fitness: float.
            Value of fitness function.
        """
        _n = len(state)
        _t = np.ceil(self.t_pct*_n)

        # Calculate head and tail values
        tail_0 = self.tail(0, state)
        head_1 = self.head(1, state)

        # Calculate R(X, T)
        if (tail_0 > _t and head_1 > _t):
            _r = _n
        else:
            _r = 0

        # Evaluate function
        fitness = max(tail_0, head_1) + _r

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