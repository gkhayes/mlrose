""" Classes for defining decay schedules for simulated annealing."""

# Author: Genevieve Hayes
# License: BSD 3 clause


class CustomSchedule:
    """Class for generating your own temperature schedule.

    Parameters
    ----------
    schedule: callable
        Function for calculating the temperature at time t with the signature
        :code:`schedule(t, **kwargs)`.

    kwargs: additional arguments
        Additional parameters to be passed to schedule.

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose_hiive
        >>> def custom(t, c): return t + c
        >>> kwargs = {'c': 10}
        >>> schedule = mlrose_hiive.CustomSchedule(custom, **kwargs)
        >>> schedule.evaluate(5)
        15
    """

    def __init__(self, schedule, **kwargs):

        self.schedule = schedule
        self.kwargs = kwargs

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

        temp = self.schedule(t, **self.kwargs)
        return temp

    def get_info__(self, t=None, prefix=''):
        prefix = f'_{prefix}__schedule_' if len(prefix) > 0 else 'schedule_'
        info = {
            f'{prefix}type': 'custom',
            f'{prefix}schedule': self.schedule
        }
        info.update({f'{prefix}_args_{k}': v for k, v in self.kwargs.items()})
        if t is not None:
            info[f'{prefix}current_value'] = self.evaluate(t)
        return info

    def __str__(self):
        return str(self.schedule)

    def __repr__(self):
        return f'{self.__class__.__name__}[{self.__dict__}]'
