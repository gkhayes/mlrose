""" Classes for defining neural network weight optimization problems."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause


import numpy as np
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod

from mlrose import GeomDecay, random_hill_climb, simulated_annealing, genetic_alg
from mlrose.opt_probs import ContinuousOpt
from . import NetworkWeights
from .activation import (identity, relu, sigmoid, tanh)
from .utils import (unflatten_weights)
from .gradient_descent import (gradient_descent)


class BaseNeuralNetwork(BaseEstimator, ABC):
    """Base class for neural networks.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self, hidden_nodes=None,
                 activation='relu',
                 algorithm='random_hill_climb',
                 max_iters=100,
                 bias=True,
                 is_classifier=True,
                 learning_rate=0.1,
                 early_stopping=False,
                 clip_max=1e+10,
                 restarts=0,
                 schedule=GeomDecay(),
                 pop_size=200,
                 mutation_prob=0.1,
                 max_attempts=10,
                 random_state=None,
                 curve=False):

        if hidden_nodes is None:
            self.hidden_nodes = []
        else:
            self.hidden_nodes = hidden_nodes

        self.activation_dict = {'identity': identity,
                                'relu': relu,
                                'sigmoid': sigmoid,
                                'tanh': tanh}
        self.activation = activation
        self.algorithm = algorithm
        self.max_iters = max_iters
        self.bias = bias
        self.is_classifier = is_classifier
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.clip_max = clip_max
        self.restarts = restarts
        self.schedule = schedule
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.max_attempts = max_attempts
        self.random_state = random_state
        self.curve = curve

        self.node_list = []
        self.fitted_weights = []
        self.loss = np.inf
        self.output_activation = None
        self.predicted_probs = []
        self.fitness_curve = []

    def _validate(self):
        if (not isinstance(self.max_iters, int) and self.max_iters != np.inf
                and not self.max_iters.is_integer()) or (self.max_iters < 0):
            raise Exception("""max_iters must be a positive integer.""")

        if not isinstance(self.bias, bool):
            raise Exception("""bias must be True or False.""")

        if not isinstance(self.is_classifier, bool):
            raise Exception("""is_classifier must be True or False.""")

        if self.learning_rate <= 0:
            raise Exception("""learning_rate must be greater than 0.""")

        if not isinstance(self.early_stopping, bool):
            raise Exception("""early_stopping must be True or False.""")

        if self.clip_max <= 0:
            raise Exception("""clip_max must be greater than 0.""")

        if (not isinstance(self.max_attempts, int) and not
                self.max_attempts.is_integer()) or (self.max_attempts < 0):
            raise Exception("""max_attempts must be a positive integer.""")

        if self.pop_size < 0:
            raise Exception("""pop_size must be a positive integer.""")
        elif not isinstance(self.pop_size, int):
            if self.pop_size.is_integer():
                self.pop_size = int(self.pop_size)
            else:
                raise Exception("""pop_size must be a positive integer.""")

        if (self.mutation_prob < 0) or (self.mutation_prob > 1):
            raise Exception("""mutation_prob must be between 0 and 1.""")

        if self.activation is None or \
           self.activation not in self.activation_dict.keys():
            raise Exception("""Activation function must be one of: 'identity',
                    'relu', 'sigmoid' or 'tanh'.""")

        if self.algorithm not in ['random_hill_climb', 'simulated_annealing',
                                  'genetic_alg', 'gradient_descent']:
            raise Exception("""Algorithm must be one of: 'random_hill_climb',
                    'simulated_annealing', 'genetic_alg',
                    'gradient_descent'.""")

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
        self._validate()

        # Make sure y is an array and not a list
        y = np.array(y)

        # Convert y to 2D if necessary
        if len(np.shape(y)) == 1:
            y = np.reshape(y, [len(y), 1])

        # Verify X and y are the same length
        if not np.shape(X)[0] == np.shape(y)[0]:
            raise Exception('The length of X and y must be equal.')

        # Determine number of nodes in each layer
        input_nodes = np.shape(X)[1] + self.bias
        output_nodes = np.shape(y)[1]
        node_list = [input_nodes] + self.hidden_nodes + [output_nodes]

        num_nodes = 0

        for i in range(len(node_list) - 1):
            num_nodes += node_list[i]*node_list[i+1]

        if init_weights is not None and len(init_weights) != num_nodes:
            raise Exception("""init_weights must be None or have length %d"""
                            % (num_nodes,))

        # Set random seed
        if isinstance(self.random_state, int) and self.random_state > 0:
            np.random.seed(self.random_state)

        # Initialize optimization problem
        fitness = NetworkWeights(X, y, node_list,
                                 self.activation_dict[self.activation],
                                 self.bias, self.is_classifier,
                                 learning_rate=self.learning_rate)

        problem = ContinuousOpt(num_nodes, fitness, maximize=False,
                                min_val=-1*self.clip_max,
                                max_val=self.clip_max, step=self.learning_rate)

        if self.algorithm == 'random_hill_climb':
            fitness_curve, fitted_weights, loss = self.__run_with_rhc(init_weights, num_nodes, problem)

        elif self.algorithm == 'simulated_annealing':
            fitness_curve, fitted_weights, loss = self._run_with_sa(init_weights, num_nodes, problem)
        elif self.algorithm == 'genetic_alg':
            fitness_curve, fitted_weights, loss = self._run_with_ga(problem)
        else:  # Gradient descent case
            fitness_curve, fitted_weights, loss = self._run_with_gd(init_weights, num_nodes, problem)

        # Save fitted weights and node list
        self.node_list = node_list
        self.fitted_weights = fitted_weights
        self.loss = loss
        self.output_activation = fitness.get_output_activation()

        if self.curve:
            self.fitness_curve = fitness_curve

        return self

    def _run_with_gd(self, init_weights, num_nodes, problem):
        if init_weights is None:
            init_weights = np.random.uniform(-1, 1, num_nodes)
        fitness_curve = []
        if self.curve:
            fitted_weights, loss, fitness_curve = gradient_descent(
                problem,
                max_attempts=self.max_attempts if self.early_stopping else
                self.max_iters,
                max_iters=self.max_iters,
                curve=self.curve,
                init_state=init_weights)

        else:
            fitted_weights, loss = gradient_descent(
                problem,
                max_attempts=self.max_attempts if self.early_stopping else
                self.max_iters,
                max_iters=self.max_iters,
                curve=self.curve,
                init_state=init_weights)
        return fitness_curve, fitted_weights, loss

    def _run_with_ga(self, problem):
        fitness_curve = []
        if self.curve:
            fitted_weights, loss, fitness_curve = genetic_alg(
                problem,
                pop_size=self.pop_size,
                mutation_prob=self.mutation_prob,
                max_attempts=self.max_attempts if self.early_stopping else
                self.max_iters,
                max_iters=self.max_iters,
                curve=self.curve)
        else:
            fitted_weights, loss = genetic_alg(
                problem,
                pop_size=self.pop_size, mutation_prob=self.mutation_prob,
                max_attempts=self.max_attempts if self.early_stopping else
                self.max_iters,
                max_iters=self.max_iters,
                curve=self.curve)
        return fitness_curve, fitted_weights, loss

    def _run_with_sa(self, init_weights, num_nodes, problem):
        fitness_curve = []
        if init_weights is None:
            init_weights = np.random.uniform(-1, 1, num_nodes)
        if self.curve:
            fitted_weights, loss, fitness_curve = simulated_annealing(
                problem,
                schedule=self.schedule,
                max_attempts=self.max_attempts if self.early_stopping else
                self.max_iters,
                max_iters=self.max_iters,
                init_state=init_weights,
                curve=self.curve)
        else:
            fitted_weights, loss = simulated_annealing(
                problem,
                schedule=self.schedule,
                max_attempts=self.max_attempts if self.early_stopping else
                self.max_iters,
                max_iters=self.max_iters,
                init_state=init_weights,
                curve=self.curve)
        return fitness_curve, fitted_weights, loss

    def __run_with_rhc(self, init_weights, num_nodes, problem):
        fitness_curve = []
        fitted_weights = None
        loss = np.inf
        # Can't use restart feature of random_hill_climb function, since
        # want to keep initial weights in the range -1 to 1.
        for _ in range(self.restarts + 1):
            if init_weights is None:
                init_weights = np.random.uniform(-1, 1, num_nodes)

            if self.curve:
                current_weights, current_loss, fitness_curve = \
                    random_hill_climb(problem,
                                      max_attempts=self.max_attempts if
                                      self.early_stopping else
                                      self.max_iters,
                                      max_iters=self.max_iters,
                                      restarts=0, init_state=init_weights,
                                      curve=self.curve)
            else:
                current_weights, current_loss = random_hill_climb(
                    problem,
                    max_attempts=self.max_attempts if self.early_stopping
                    else self.max_iters,
                    max_iters=self.max_iters,
                    restarts=0, init_state=init_weights, curve=self.curve)

            if current_loss < loss:
                fitted_weights = current_weights
                loss = current_loss
        return fitness_curve, fitted_weights, loss

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
        if not np.shape(X)[1] == (self.node_list[0] - self.bias):
            raise Exception("""The number of columns in X must equal %d"""
                            % ((self.node_list[0] - self.bias),))

        weights = unflatten_weights(self.fitted_weights, self.node_list)

        # Add bias column to inputs matrix, if required
        if self.bias:
            ones = np.ones([np.shape(X)[0], 1])
            inputs = np.hstack((X, ones))

        else:
            inputs = X

        # Pass data through network
        for i in range(len(weights)):
            # Multiple inputs by weights
            outputs = np.dot(inputs, weights[i])

            # Transform outputs to get inputs for next layer (or final preds)
            if i < len(weights) - 1:
                inputs = self.activation_dict[self.activation](outputs)
            else:
                y_pred = self.output_activation(outputs)

        # For classifier, convert predicted probabilities to 0-1 labels
        if self.is_classifier:
            self.predicted_probs = y_pred

            if self.node_list[-1] == 1:
                y_pred = np.round(y_pred).astype(int)
            else:
                zeros = np.zeros_like(y_pred)
                zeros[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)] = 1
                y_pred = zeros.astype(int)

        return y_pred

    def get_params(self, deep=False):
        """Get parameters for this estimator.

        Returns
        -------
        params : dictionary
            Parameter names mapped to their values.
        """
        params = {'hidden_nodes': self.hidden_nodes,
                  'max_iters': self.max_iters,
                  'bias': self.bias,
                  'is_classifier': self.is_classifier,
                  'learning_rate': self.learning_rate,
                  'early_stopping': self.early_stopping,
                  'clip_max': self.clip_max,
                  'restarts': self.restarts,
                  'schedule': self.schedule,
                  'pop_size': self.pop_size,
                  'mutation_prob': self.mutation_prob}

        return params

    def set_params(self, **in_params):
        """Set the parameters of this estimator.

        Parameters
        -------
        in_params: dictionary
            Dictionary of parameters to be set and the value to be set to.
        """
        if 'hidden_nodes' in in_params.keys():
            self.hidden_nodes = in_params['hidden_nodes']
        if 'max_iters' in in_params.keys():
            self.max_iters = in_params['max_iters']
        if 'bias' in in_params.keys():
            self.bias = in_params['bias']
        if 'is_classifier' in in_params.keys():
            self.is_classifier = in_params['is_classifier']
        if 'learning_rate' in in_params.keys():
            self.learning_rate = in_params['learning_rate']
        if 'early_stopping' in in_params.keys():
            self.early_stopping = in_params['early_stopping']
        if 'clip_max' in in_params.keys():
            self.clip_max = in_params['clip_max']
        if 'restarts' in in_params.keys():
            self.restarts = in_params['restarts']
        if 'schedule' in in_params.keys():
            self.schedule = in_params['schedule']
        if 'pop_size' in in_params.keys():
            self.pop_size = in_params['pop_size']
        if 'mutation_prob' in in_params.keys():
            self.mutation_prob = in_params['mutation_prob']