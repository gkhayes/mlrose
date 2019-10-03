""" Neural network activation functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np


def relu(x, deriv=False):
    """ReLU activation function

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
    fx = np.copy(x)
    fx[np.where(fx < 0)] = 0

    if deriv:
        fx[np.where(fx > 0)] = 1

    return fx