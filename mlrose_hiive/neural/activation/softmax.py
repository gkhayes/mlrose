""" Neural network activation functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause
from mlrose_hiive.decorators import short_name

import numpy as np

import warnings
warnings.filterwarnings("ignore")


@short_name('softmax')
def softmax(x):
    """Softmax activation function

    Parameters
    ----------
    x: array
        Array containing input data.

    Returns
    -------
    fx: array
        Value of activation function at x
    """
    max_prob = np.max(x, axis=1).reshape((-1, 1))
    fx = np.exp(x - max_prob)
    sum_prob = np.sum(fx, axis=1).reshape((-1, 1))
    fx = np.divide(fx, sum_prob)

    return fx
