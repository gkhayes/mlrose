""" Classes for defining decay schedules for simulated annealing.

    Author: Genevieve Hayes
    License: 3-clause BSD license.
"""
import numpy as np


class GeomDecay:
    """
    Schedule for geometrically decaying the simulated
    annealing temperature parameter T
    """

    def __init__(self, init_temp=1.0, decay=0.99, min_temp=0.001):
        """Initialize decay schedule object

        Args:
        init_temp: float. Initial value of temperature parameter T
        decay: float. Temperature decay parameter.
        min_temp: float. Minimum value of temperature parameter.

        Returns:
        None
        """
        self.init_temp = init_temp
        self.decay = decay
        self.min_temp = min_temp

        if self.init_temp <= 0:
            raise Exception("""init_temp must be greater than 0.""")

        if (self.decay <= 0) or (self.decay > 1):
            raise Exception("""decay must be between 0 and 1.""")

        if self.min_temp < 0:
            raise Exception("""min_temp must be greater than 0.""")
        elif self.min_temp > self.init_temp:
            raise Exception("""init_temp must be greater than min_temp.""")

    def evaluate(self, _t):
        """Evaluate the temperature parameter at time t

        Args:
        t: int. Time at which the temperature paramter t is evaluated

        Returns:
        temp: float. Temperature parameter at time t
        """

        temp = self.init_temp*(self.decay**_t)

        if temp < self.min_temp:
            temp = self.min_temp

        return temp


class ArithDecay:
    """
    Schedule for arithmetically decaying the simulated
    annealing temperature parameter T
    """

    def __init__(self, init_temp=1.0, decay=0.0001, min_temp=0.001):
        """Initialize decay schedule object

        Args:
        init_temp: float. Initial value of temperature parameter T
        decay: float. Temperature decay parameter.
        min_temp: float. Minimum value of temperature parameter.

        Returns:
        None
        """
        self.init_temp = init_temp
        self.decay = decay
        self.min_temp = min_temp

        if self.init_temp <= 0:
            raise Exception("""init_temp must be greater than 0.""")

        if (self.decay <= 0) or (self.decay > 1):
            raise Exception("""decay must be greater than 0.""")

        if self.min_temp < 0:
            raise Exception("""min_temp must be greater than 0.""")
        elif self.min_temp > self.init_temp:
            raise Exception("""init_temp must be greater than min_temp.""")

    def evaluate(self, _t):
        """Evaluate the temperature parameter at time t

        Args:
        t: int. Time at which the temperature paramter t is evaluated

        Returns:
        temp: float. Temperature parameter at time t
        """

        temp = self.init_temp - (self.decay*_t)

        if temp < self.min_temp:
            temp = self.min_temp

        return temp


class ExpDecay:
    """
    Schedule for exponentially decaying the simulated
    annealing temperature parameter T
    """

    def __init__(self, init_temp=1.0, exp_const=0.005, min_temp=0.001):
        """Initialize decay schedule object

        Args:
        init_temp: float. Initial value of temperature parameter T
        exp_const: float. Exponential constant parameter.
        min_temp: float. Minimum value of temperature parameter.

        Returns:
        None
        """
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

    def evaluate(self, _t):
        """Evaluate the temperature parameter at time t

        Args:
        t: int. Time at which the temperature paramter t is evaluated

        Returns:
        temp: float. Temperature parameter at time t
        """

        temp = self.init_temp*np.exp(-1.0*self.exp_const*_t)

        if temp < self.min_temp:
            temp = self.min_temp

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
