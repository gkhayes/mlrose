""" Neural network activation functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np


def identity(x, deriv=False):
    """Linear activation function

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
    if not deriv:
        fx = x
    else:
        fx = np.ones(np.shape(x))

    return fx
