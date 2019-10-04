""" Classes for defining neural network weight optimization problems."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np


def flatten_weights(weights):
    """Flatten list of weights arrays into a 1D array.

    Parameters
    ----------
    weights: list of arrays
        List of 2D arrays for flattening.

    Returns
    -------
    flat_weights: array
        1D weights array.
    """
    flat_weights = []

    for i in range(len(weights)):
        flat_weights += list(weights[i].flatten())

    flat_weights = np.array(flat_weights)

    return flat_weights


def unflatten_weights(flat_weights, node_list):
    """Convert 1D weights array into list of 2D arrays.

    Parameters
    ----------
    flat_weights: array
        1D weights array.

    node_list: list
        List giving the number of nodes in each layer of the network,
        including the input and output layers.

    Returns
    -------
    weights: list of arrays
        List of 2D arrays created from flat_weights.
    """
    nodes = 0
    for i in range(len(node_list) - 1):
        nodes += node_list[i]*node_list[i + 1]

    if len(flat_weights) != nodes:
        raise Exception("""flat_weights must have length %d""" % (nodes,))

    weights = []
    start = 0

    for i in range(len(node_list) - 1):
        end = start + node_list[i]*node_list[i + 1]
        weights.append(np.reshape(flat_weights[start:end],
                                  [node_list[i], node_list[i+1]]))
        start = end

    return weights
