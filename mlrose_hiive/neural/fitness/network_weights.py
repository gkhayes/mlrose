""" Classes for defining neural network weight optimization problems."""

# Author: Genevieve Hayes
# License: BSD 3 clause

import numpy as np
import sklearn.metrics as skm

from mlrose_hiive.neural.utils import unflatten_weights
from mlrose_hiive.neural import activation as act


class NetworkWeights:
    """Fitness function for neural network weights optimization problem.

    Parameters
    ----------
    X: array
        Numpy array containing feature dataset with each row representing a
        single observation.

    y: array
        Numpy array containing true values of data labels.
        Length must be same as length of X.

    node_list: list of ints
        Number of nodes in each layer, including the input and output layers.

    activation: callable
        Activation function for each of the hidden layers with the signature
        :code:`activation(x, deriv)`, where setting deriv is a boolean that
        determines whether to return the activation function or its derivative.

    bias: bool, default: True
        Whether a bias term is included in the network.

    is_classifer: bool, default: True
        Whether the network is for classification or regression. Set True for
        classification and False for regression.
    """

    def __init__(self, X, y, node_list, activation, bias=True,
                 is_classifier=True, learning_rate=0.1):

        # Make sure y is an array and not a list
        y = np.array(y)

        # Convert y to 2D if necessary
        if len(np.shape(y)) == 1:
            y = np.reshape(y, [len(y), 1])

        # Verify X and y are the same length
        if not np.shape(X)[0] == np.shape(y)[0]:
            raise Exception("""The length of X and y must be equal.""")

        if len(node_list) < 2:
            raise Exception("""node_list must contain at least 2 elements.""")

        if not np.shape(X)[1] == (node_list[0] - bias):
            raise Exception("""The number of columns in X must equal %d"""
                            % ((node_list[0] - bias),))

        if not np.shape(y)[1] == node_list[-1]:
            raise Exception("""The number of columns in y must equal %d"""
                            % (node_list[-1],))

        if not isinstance(bias, bool):
            raise Exception("""bias must be True or False.""")

        if not isinstance(is_classifier, bool):
            raise Exception("""is_classifier must be True or False.""")

        if learning_rate <= 0:
            raise Exception("""learning_rate must be greater than 0.""")

        self.X = X
        self.y_true = y
        self.node_list = node_list
        self.activation = activation
        self.bias = bias
        self.is_classifier = is_classifier
        self.learning_rate = learning_rate

        # Determine appropriate loss function and output activation function
        if self.is_classifier:
            self.loss = skm.log_loss

            if np.shape(self.y_true)[1] == 1:
                self.output_activation = act.sigmoid
            else:
                self.output_activation = act.softmax
        else:
            self.loss = skm.mean_squared_error
            self.output_activation = act.identity

        self.inputs_list = []
        self.y_pred = y
        self.weights = []
        self.prob_type = 'continuous'

        nodes = 0
        for i in range(len(node_list) - 1):
            nodes += node_list[i]*node_list[i + 1]

        self.nodes = nodes

    def evaluate(self, state):
        """Evaluate the fitness of a state.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """
        if not len(state) == self.nodes:
            raise Exception("""state must have length %d""" % (self.nodes,))

        self.inputs_list = []
        self.weights = unflatten_weights(state, self.node_list)

        # Add bias column to inputs matrix, if required
        if self.bias:
            ones = np.ones([np.shape(self.X)[0], 1])
            inputs = np.hstack((self.X, ones))

        else:
            inputs = self.X

        # Pass data through network
        for i in range(len(self.weights)):
            # Multiple inputs by weights
            outputs = np.dot(inputs, self.weights[i])
            self.inputs_list.append(inputs)

            # Transform outputs to get inputs for next layer (or final preds)
            if i < len(self.weights) - 1:
                inputs = self.activation(outputs)
            else:
                self.y_pred = self.output_activation(outputs)

        # Evaluate loss function
        fitness = self.loss(self.y_true, self.y_pred)

        return fitness

    def get_output_activation(self):
        """ Return the activation function for the output layer.

        Returns
        -------
        self.output_activation: callable
            Activation function for the output layer.
        """
        return self.output_activation

    def get_prob_type(self):
        """ Return the problem type.

        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp', or
            'either'.
        """
        return self.prob_type

    def calculate_updates(self):
        """Calculate gradient descent updates.

        Returns
        -------
        updates_list: list
            List of back propagation weight updates.
        """
        delta_list = []
        updates_list = []

        # Work backwards from final layer
        for i in range(len(self.inputs_list)-1, -1, -1):
            # Final layer
            if i == len(self.inputs_list)-1:
                delta = (self.y_pred - self.y_true)
            # Hidden layers
            else:
                dot = np.dot(delta_list[-1], np.transpose(self.weights[i+1]))
                activation = self.activation(self.inputs_list[i+1], deriv=True)
                delta = dot * activation

            delta_list.append(delta)

            # Calculate updates
            updates = (-1.0 * self.learning_rate *
                       np.dot(np.transpose(self.inputs_list[i]), delta))

            updates_list.append(updates)

        # Reverse order of updates list
        updates_list = updates_list[::-1]

        return updates_list
