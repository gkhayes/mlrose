""" Classes for defining neural network weight optimization problems."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause


import numpy as np
from abc import abstractmethod
from mlrose_hiive.algorithms.decay import GeomDecay
from mlrose_hiive.algorithms.rhc import random_hill_climb
from mlrose_hiive.algorithms.sa import simulated_annealing
from mlrose_hiive.algorithms.ga import  genetic_alg

from mlrose_hiive.neural._nn_base import _NNBase
from mlrose_hiive.neural.activation import (identity, relu, sigmoid, tanh)
from mlrose_hiive.neural.utils.weights import gradient_descent_original


class _NNCore(_NNBase):
    """Core class for neural networks.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self, hidden_nodes=None, activation='relu', algorithm='random_hill_climb', max_iters=100, bias=True,
                 is_classifier=True, learning_rate=0.1, early_stopping=False, clip_max=1e+10, restarts=0,
                 schedule=GeomDecay(), pop_size=200, mutation_prob=0.1, max_attempts=10, random_state=None,
                 curve=False):

        super().__init__()
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

        X, y = self._format_x_y_data(X, y)

        node_list = self._build_node_list(X, y, self.hidden_nodes, self.bias)

        num_nodes = self._calculate_state_size(node_list)

        if init_weights is not None and len(init_weights) != num_nodes:
            raise Exception("""init_weights must be None or have length %d"""
                            % (num_nodes,))

        # Set random seed
        if isinstance(self.random_state, int) and self.random_state > 0:
            np.random.seed(self.random_state)

        fitness, problem = self._build_problem_and_fitness_function(X, y,
                                                                    node_list,
                                                                    self.activation_dict[self.activation],
                                                                    self.learning_rate,
                                                                    self.bias,
                                                                    self.clip_max,
                                                                    self.is_classifier)

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

        fitted_weights, loss, fitness_curve = gradient_descent_original(
            problem,
            max_attempts=self.max_attempts if self.early_stopping else self.max_iters,
            max_iters=self.max_iters,
            curve=self.curve,
            init_state=init_weights)

        return ([] if fitness_curve is None else fitness_curve), fitted_weights, loss

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
            fitted_weights, loss, _ = genetic_alg(
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
            fitted_weights, loss, _ = simulated_annealing(
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
        fitted_weights = []
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
                current_weights, current_loss, _ = random_hill_climb(
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

        y_pred, pp = self._predict(X=X,
                                   fitted_weights=self.fitted_weights,
                                   node_list=self.node_list,
                                   input_activation=self.activation_dict[self.activation],
                                   output_activation=self.output_activation,
                                   bias=self.bias,
                                   is_classifier=self.is_classifier)
        self.predicted_probs = pp
        return y_pred
