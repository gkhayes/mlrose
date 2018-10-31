""" Classes for defining decay schedules for simulated annealing.

    Author: Genevieve Hayes <ghayes17@gmail.com>
    License: 3-clause BSD license.
"""

import numpy as np


class GeomDecay:
    """
    Schedule for geometrically decaying the simulated
    annealing temperature parameter T
    """

    def __init__(self, init_t=1.0, decay=0.99, min_t=0.001):
        """Initialize decay schedule object

        Args:
        init_t: float. Initial value of temperature parameter T
        decay: float. Temperature decay parameter.
        min_t: float. Minimum value of temperature parameter.

        Returns:
        None
        """
        self.init_t = init_t
        self.decay = decay
        self.min_t = min_t

    def evaluate(self, _t):
        """Evaluate the temperature parameter at time t

        Args:
        t: int. Time at which the temperature paramter t is evaluated

        Returns:
        temp: float. Temperature parameter at time t
        """

        temp = self.init_t*(self.decay**_t)

        if temp < self.min_t:
            temp = self.min_t

        return temp


class ArithDecay:
    """
    Schedule for arithmetically decaying the simulated
    annealing temperature parameter T
    """

    def __init__(self, init_t=1.0, decay=0.0001, min_t=0.001):
        """Initialize decay schedule object

        Args:
        init_t: float. Initial value of temperature parameter T
        decay: float. Temperature decay parameter.
        min_t: float. Minimum value of temperature parameter.

        Returns:
        None
        """
        self.init_t = init_t
        self.decay = decay
        self.min_t = min_t

    def evaluate(self, _t):
        """Evaluate the temperature parameter at time t

        Args:
        t: int. Time at which the temperature paramter t is evaluated

        Returns:
        temp: float. Temperature parameter at time t
        """

        temp = self.init_t - (self.decay*_t)

        if temp < self.min_t:
            temp = self.min_t

        return temp


class ExpDecay:
    """
    Schedule for exponentially decaying the simulated
    annealing temperature parameter T
    """

    def __init__(self, init_t=1.0, exp_const=0.005, min_t=0.001):
        """Initialize decay schedule object

        Args:
        init_t: float. Initial value of temperature parameter T
        exp_const: float. Exponential constant parameter.
        min_t: float. Minimum value of temperature parameter.

        Returns:
        None
        """
        self.init_t = init_t
        self.exp_const = exp_const
        self.min_t = min_t

    def evaluate(self, _t):
        """Evaluate the temperature parameter at time t

        Args:
        t: int. Time at which the temperature paramter t is evaluated

        Returns:
        temp: float. Temperature parameter at time t
        """

        temp = self.init_t*np.exp(-1.0*self.exp_const*_t)

        if temp < self.min_t:
            temp = self.min_t

        return temp


class CustomSchedule:
    """Class for generating your own temperature schedule."""

    def __init__(self, schedule, **kwargs):
        """Initialize CustomSchedule object.

        Args:
        schedule: function. Function for calculating the temperature at time t
        kwargs: dictionary. Additional arguments to be passed to schedule

        Returns:
        None
        """
        self.schedule = schedule
        self.kwargs = kwargs

    def evaluate(self, _t):
        """Evaluate the temperature parameter at time t

        Args:
        t: int. Time at which the temperature paramter t is evaluated

        Returns:
        temp: float. Temperature parameter at time t
        """
        temp = self.schedule(_t, **self.kwargs)
        return temp
