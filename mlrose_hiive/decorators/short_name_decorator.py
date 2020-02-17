""" Functions to implement the randomized optimization and search algorithms.
"""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause


def short_name(expr):
    def short_name_func_applicator(func):
        func.__short_name__ = expr
        return func
    return short_name_func_applicator


def get_short_name(v):
    return v if not hasattr(v, '__short_name__') else v.__short_name__
