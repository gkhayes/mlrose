""" Classes for defining neural network weight optimization problems."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause
from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

from mlrose_hiive.neural.fitness.network_weights import NetworkWeights
from mlrose_hiive.neural.utils import (unflatten_weights)
from mlrose_hiive.opt_probs import ContinuousOpt


class _NNBase(BaseEstimator, ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y=None, init_weights=None):
        """Fit neural network to data.

        Parameters
        ----------
        X: array
            Numpy array containing feature dataset with each row
            representing a single observation.

        y: array
            Numpy array containing data labels. Length must be same as
            length of X.

        init_state: array, default: None
            Numpy array containing starting weights for algorithm.
            If :code:`None`, then a random state is used.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Use model to predict data labels for given feature array.

        Parameters
        ----------
        X: array
            Numpy array containing feature dataset with each row
            representing a single observation.

        Returns
        -------
        y_pred: array
            Numpy array containing predicted data labels.
        """
        pass

    @staticmethod
    def _calculate_state_size(node_list):
        num_nodes = 0
        for i in range(len(node_list) - 1):
            num_nodes += node_list[i] * node_list[i + 1]
        return num_nodes

    @staticmethod
    def _build_node_list(X, y, hidden_nodes, bias):
        # Determine number of nodes in each layer
        input_nodes = np.shape(X)[1] + bias
        output_nodes = np.shape(y)[1]
        node_list = [input_nodes] + list(hidden_nodes) + [output_nodes]
        return node_list

    @staticmethod
    def _format_x_y_data(X, y):
        # Make sure y is an array and not a list
        y = np.array(y)
        # Convert y to 2D if necessary
        if len(np.shape(y)) == 1:
            y = np.reshape(y, [len(y), 1])
        # Verify X and y are the same length
        if not np.shape(X)[0] == np.shape(y)[0]:
            raise Exception('The length of X and y must be equal.')
        return X, y

    @staticmethod
    def _build_problem_and_fitness_function(X, y, node_list, activation, learning_rate,
                                            bias, clip_max, is_classifier=True):
        # Initialize optimization problem
        fitness = NetworkWeights(X, y, node_list,
                                 activation,
                                 bias, is_classifier,
                                 learning_rate=learning_rate)
        num_nodes = _NNBase._calculate_state_size(node_list)
        problem = ContinuousOpt(length=num_nodes,
                                fitness_fn=fitness,
                                maximize=False,
                                min_val=-1 * clip_max,
                                max_val=clip_max,
                                step=learning_rate)
        return fitness, problem

    @staticmethod
    def _predict(X, fitted_weights, node_list, bias, input_activation, output_activation,
                 is_classifier=True):

        weights = unflatten_weights(fitted_weights, node_list)

        # Add bias column to inputs matrix, if required
        if bias:
            ones = np.ones([np.shape(X)[0], 1])
            inputs = np.hstack((X, ones))

        else:
            inputs = X

        # Pass data through network
        y_pred = None
        for i in range(len(weights)):
            # Multiple inputs by weights
            outputs = np.dot(inputs, weights[i])

            # Transform outputs to get inputs for next layer (or final preds)
            if i < len(weights) - 1:
                inputs = input_activation(outputs)
            else:
                y_pred = output_activation(outputs)

        # For classifier, convert predicted probabilities to 0-1 labels
        predicted_probs = None
        if is_classifier:
            predicted_probs = y_pred

            if node_list[-1] == 1:
                y_pred = np.round(y_pred).astype(int)
            else:
                zeros = np.zeros_like(y_pred)
                zeros[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)] = 1
                y_pred = zeros.astype(int)

        return y_pred, predicted_probs
