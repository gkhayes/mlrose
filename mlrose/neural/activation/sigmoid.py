""" Neural network activation functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np


def sigmoid(x, deriv=False):
    """Sigmoid activation function

    Parameters
    ----------
    x: array
        Array containing input data.

    deriv: bool, default: False
        Whether to return the function or its derivative.
        Set True for derivative.

    Returns
    -------
    fx: array
        Value of activation function at x
    """
    fx = 1/(1 + np.exp(-x))

    if deriv:
        fx *= (1 - fx)

    return fx
