""" Neural network activation functions."""

# Author: Genevieve Hayes
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
    fx = np.exp(x)/np.reshape(np.sum(np.exp(x), axis=1), [len(x), 1])

    return fx


def tanh(x, deriv=False):
    """Hyperbolic tan activation function

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
    fx = np.tanh(x)

    if deriv:
        fx = 1 - fx**2

    return fx
