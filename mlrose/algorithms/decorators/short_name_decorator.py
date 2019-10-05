""" Functions to implement the randomized optimization and search algorithms.
"""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause


def short_name(expr):
    def short_name_func_applicator(func):
        func.__short_name__ = expr
        return func
    return short_name_func_applicator
