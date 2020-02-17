""" Classes for defining decay schedules for simulated annealing."""

# Author: Genevieve Hayes
# License: BSD 3 clause

import numpy as np


class ExpDecay:
    """
    Schedule for exponentially decaying the simulated
    annealing temperature parameter T according to the formula:

    .. math::

        T(t) = \\max(T_{0} e^{-rt}, T_{min})

    where:

    * :math:`T_{0}` is the initial temperature (at time t = 0);
    * :math:`r` is the rate of exponential decay; and
    * :math:`T_{min}` is the minimum temperature value.

    Parameters
    ----------
    init_temp: float, default: 1.0
        Initial value of temperature parameter T. Must be greater than 0.
    exp_const: float, default: 0.005
        Exponential constant parameter, r. Must be greater than 0.
    min_temp: float, default: 0.001
        Minimum value of temperature parameter. Must be greater than 0.

    Example
    -------
    .. highlight:: python
    .. code-block:: python

       >>> import mlrose_hiive
       >>> schedule = mlrose_hiive.ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)
       >>> schedule.evaluate(5)
       7.78800...
    """

    def __init__(self, init_temp=1.0, exp_const=0.005, min_temp=0.001):

        self.init_temp = init_temp
        self.exp_const = exp_const
        self.min_temp = min_temp

        if self.init_temp <= 0:
            raise Exception("""init_temp must be greater than 0.""")

        if self.exp_const <= 0:
            raise Exception("""exp_const must be greater than 0.""")

        if self.min_temp < 0:
            raise Exception("""min_temp must be greater than 0.""")
        elif self.min_temp > self.init_temp:
            raise Exception("""init_temp must be greater than min_temp.""")

    def evaluate(self, t):
        """Evaluate the temperature parameter at time t.

        Parameters
        ----------
        t: int
            Time at which the temperature paramter T is evaluated.

        Returns
        -------
        temp: float
            Temperature parameter at time t.
        """

        temp = self.init_temp*np.exp(-1.0*self.exp_const*t)

        if temp < self.min_temp:
            temp = self.min_temp

        return temp

    def get_info__(self, t=None, prefix=''):
        prefix = f'_{prefix}__schedule_' if len(prefix) > 0 else 'schedule_'
        info = {
            f'{prefix}type': 'exponential',
            f'{prefix}init_temp': self.init_temp,
            f'{prefix}exp_const': self.exp_const,
            f'{prefix}min_temp': self.min_temp,
        }
        if t is not None:
            info[f'{prefix}current_value'] = self.evaluate(t)
        return info

    def __str__(self):
        return str(self.init_temp)

    def __repr__(self):
        return f'{self.__class__.__name__}(init_temp={self.init_temp}, ' \
               f'exp_const={self.exp_const}, min_temp={self.min_temp})'
