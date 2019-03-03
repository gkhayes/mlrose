""" Unit tests for neural.py"""

# Author: Genevieve Hayes
# License: BSD 3 clause

import unittest
import numpy as np
# The following functions/classes are not automatically imported at
# initialization, so must be imported explicitly from neural.py and
# activation.py.
from mlrose.neural import (flatten_weights, unflatten_weights,
                           gradient_descent, NetworkWeights, ContinuousOpt,
                           NeuralNetwork, LogisticRegression, LinearRegression)
from mlrose.activation import identity, sigmoid, softmax


class TestNeural(unittest.TestCase):
    """Tests for neural.py functions."""

    @staticmethod
    def test_flatten_weights():
        """Test flatten_weights function"""

        x = np.arange(12)
        y = np.arange(6)
        z = np.arange(16)

        a = np.reshape(x, (4, 3))
        b = np.reshape(y, (3, 2))
        c = np.reshape(z, (2, 8))

        weights = [a, b, c]

        flat = list(x) + list(y) + list(z)

        assert np.array_equal(np.array(flatten_weights(weights)),
                              np.array(flat))

    @staticmethod
    def test_unflatten_weights():
        """Test unflatten_weights function"""

        x = np.arange(12)
        y = np.arange(6)
        z = np.arange(16)

        a = np.reshape(x, (4, 3))
        b = np.reshape(y, (3, 2))
        c = np.reshape(z, (2, 8))

        flat = list(x) + list(y) + list(z)
        nodes = [4, 3, 2, 8]
        weights = unflatten_weights(flat, nodes)

        assert (np.array_equal(weights[0], a)
                and np.array_equal(weights[1], b)
                and np.array_equal(weights[2], c))

    @staticmethod
    def test_gradient_descent():
        """Test gradient_descent function"""

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        nodes = [4, 2, 1]

        fitness = NetworkWeights(X, y, nodes, activation=identity,
                                 bias=False, is_classifier=False,
                                 learning_rate=0.1)

        problem = ContinuousOpt(10, fitness, maximize=False,
                                min_val=-1, max_val=1, step=0.1)

        test_weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        test_fitness = -1*problem.eval_fitness(test_weights)

        best_state, best_fitness = gradient_descent(problem)

        assert (len(best_state) == 10 and min(best_state) >= -1
                and max(best_state) <= 1 and best_fitness < test_fitness)

    @staticmethod
    def test_gradient_descent_iter1():
        """Test gradient_descent function gets the correct answer after a
        single iteration"""

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        nodes = [4, 2, 1]

        fitness = NetworkWeights(X, y, nodes, activation=identity,
                                 bias=False, is_classifier=False,
                                 learning_rate=0.1)

        problem = ContinuousOpt(10, fitness, maximize=False,
                                min_val=-1, max_val=1, step=0.1)

        init_weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        best_state, best_fitness = gradient_descent(problem, max_iters=1,
                                                    init_state=init_weights)

        x = np.array([-0.7, -0.7, -0.9, -0.9, -0.9, -0.9, -1, -1, -1, -1])

        assert (np.allclose(best_state, x, atol=0.001)
                and round(best_fitness, 2) == 19.14)


class TestNeuralWeights(unittest.TestCase):
    """Tests for NeuralWeights class."""

    @staticmethod
    def test_evaluate_no_bias_classifier():
        """Test evaluate method for binary classifier with no bias term"""

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        nodes = [4, 2, 1]

        fitness = NetworkWeights(X, y, nodes, activation=identity,
                                 bias=False)

        a = list(np.arange(8) + 1)
        b = list(0.01*(np.arange(2) + 1))

        weights = a + b

        assert round(fitness.evaluate(weights), 4) == 0.7393

    @staticmethod
    def test_evaluate_no_bias_multi():
        """Test evaluate method for multivariate classifier with no bias
        term"""

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.array([[1, 1],
                      [1, 0],
                      [0, 0],
                      [0, 0],
                      [1, 0],
                      [1, 1]])

        nodes = [4, 2, 2]

        fitness = NetworkWeights(X, y, nodes, activation=identity,
                                 bias=False)

        a = list(np.arange(8) + 1)
        b = list(0.01*(np.arange(4) + 1))

        weights = a + b

        assert round(fitness.evaluate(weights), 4) == 0.7183

    @staticmethod
    def test_evaluate_no_bias_regressor():
        """Test evaluate method for regressor with no bias term"""

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        nodes = [4, 2, 1]

        fitness = NetworkWeights(X, y, nodes, activation=identity,
                                 bias=False, is_classifier=False)

        a = list(np.arange(8) + 1)
        b = list(0.01*(np.arange(2) + 1))

        weights = a + b

        assert round(fitness.evaluate(weights), 4) == 0.5542

    @staticmethod
    def test_evaluate_bias_regressor():
        """Test evaluate method for regressor with bias term"""

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        nodes = [5, 2, 1]

        fitness = NetworkWeights(X, y, nodes, activation=identity,
                                 bias=True, is_classifier=False)

        a = list(np.arange(10) + 1)
        b = list(0.01*(np.arange(2) + 1))

        weights = a + b

        assert round(fitness.evaluate(weights), 4) == 0.4363

    @staticmethod
    def test_calculate_updates():
        """Test calculate_updates method"""

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        nodes = [4, 2, 1]

        fitness = NetworkWeights(X, y, nodes, activation=identity,
                                 bias=False, is_classifier=False,
                                 learning_rate=1)

        a = list(np.arange(8) + 1)
        b = list(0.01*(np.arange(2) + 1))

        weights = a + b
        fitness.evaluate(weights)

        updates = fitness.calculate_updates()

        update1 = np.array([[-0.0017, -0.0034],
                            [-0.0046, -0.0092],
                            [-0.0052, -0.0104],
                            [0.0014, 0.0028]])

        update2 = np.array([[-3.17],
                            [-4.18]])

        assert (np.allclose(updates[0], update1, atol=0.001)
                and np.allclose(updates[1], update2, atol=0.001))


class TestNeuralNetwork(unittest.TestCase):
    """Tests for NeuralNetwork class."""

    @staticmethod
    def test_fit_random_hill_climb():
        """Test fit method using the random hill climbing algorithm"""

        network = NeuralNetwork(hidden_nodes=[2], activation='identity',
                                algorithm='random_hill_climb',
                                bias=False, is_classifier=True,
                                learning_rate=1, clip_max=1,
                                max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        network.fit(X, y, init_weights=weights)
        fitted = network.fitted_weights

        assert (sum(fitted) < 10 and len(fitted) == 10 and min(fitted) >= -1
                and max(fitted) <= 1)

    @staticmethod
    def test_fit_simulated_annealing():
        """Test fit method using the simulated_annealing algorithm"""

        network = NeuralNetwork(hidden_nodes=[2], activation='identity',
                                algorithm='simulated_annealing',
                                bias=False, is_classifier=True,
                                learning_rate=1, clip_max=1,
                                max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        network.fit(X, y, init_weights=weights)
        fitted = network.fitted_weights

        assert (sum(fitted) < 10 and len(fitted) == 10 and min(fitted) >= -1
                and max(fitted) <= 1)

    @staticmethod
    def test_fit_genetic_alg():
        """Test fit method using the genetic_alg algorithm"""

        network = NeuralNetwork(hidden_nodes=[2], activation='identity',
                                algorithm='genetic_alg',
                                bias=False, is_classifier=True,
                                learning_rate=1, clip_max=1,
                                max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        network.fit(X, y)
        fitted = network.fitted_weights

        assert (sum(fitted) < 10 and len(fitted) == 10 and min(fitted) >= -1
                and max(fitted) <= 1)

    @staticmethod
    def test_fit_gradient_descent():
        """Test fit method using the gradient_descent algorithm"""

        network = NeuralNetwork(hidden_nodes=[2], activation='identity',
                                algorithm='gradient_descent',
                                bias=False, is_classifier=True,
                                learning_rate=1, clip_max=1,
                                max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        network.fit(X, y, init_weights=weights)
        fitted = network.fitted_weights

        assert (sum(fitted) < 10 and len(fitted) == 10 and min(fitted) >= -1
                and max(fitted) <= 1)

    @staticmethod
    def test_predict_no_bias():
        """Test predict method with no bias term"""

        network = NeuralNetwork(hidden_nodes=[2], activation='identity',
                                algorithm='random_hill_climb',
                                bias=False, is_classifier=True,
                                learning_rate=1, clip_max=1,
                                max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        network.fitted_weights = np.array([0.2, 0.5, 0.3, 0.4, 0.4, 0.3,
                                           0.5, 0.2, -1, 1, 1, -1])
        network.node_list = [4, 2, 2]
        network.output_activation = softmax

        probs = np.array([[0.40131, 0.59869],
                          [0.5, 0.5],
                          [0.5, 0.5],
                          [0.5, 0.5],
                          [0.31003, 0.68997],
                          [0.64566, 0.35434]])

        labels = np.array([[0, 1],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [0, 1],
                           [1, 0]])

        assert (np.array_equal(network.predict(X), labels)
                and np.allclose(network.predicted_probs, probs, atol=0.0001))

    @staticmethod
    def test_predict_bias():
        """Test predict method with bias term"""

        network = NeuralNetwork(hidden_nodes=[2], activation='identity',
                                algorithm='random_hill_climb',
                                bias=True, is_classifier=True,
                                learning_rate=1, clip_max=1,
                                max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        network.fitted_weights = np.array([0.2, 0.5, 0.3, 0.4, 0.4, 0.3,
                                           0.5, 0.2, 1, -1, -0.1, 0.1,
                                           0.1, -0.1])
        network.node_list = [5, 2, 2]
        network.output_activation = softmax

        probs = np.array([[0.39174, 0.60826],
                          [0.40131, 0.59869],
                          [0.40131, 0.59869],
                          [0.40131, 0.59869],
                          [0.38225, 0.61775],
                          [0.41571, 0.58419]])

        labels = np.array([[0, 1],
                           [0, 1],
                           [0, 1],
                           [0, 1],
                           [0, 1],
                           [0, 1]])

        assert (np.array_equal(network.predict(X), labels)
                and np.allclose(network.predicted_probs, probs, atol=0.0001))


class TestLinearRegression(unittest.TestCase):
    """Tests for LinearRegression class."""

    @staticmethod
    def test_fit_random_hill_climb():
        """Test fit method using the random hill climbing algorithm"""

        network = LinearRegression(algorithm='random_hill_climb', bias=False,
                                   learning_rate=1, clip_max=1,
                                   max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        weights = np.array([1, 1, 1, 1])

        network.fit(X, y, init_weights=weights)
        fitted = network.fitted_weights

        assert (sum(fitted) < 4 and len(fitted) == 4 and min(fitted) >= -1
                and max(fitted) <= 1)

    @staticmethod
    def test_fit_simulated_annealing():
        """Test fit method using the simulated_annealing algorithm"""

        network = LinearRegression(algorithm='simulated_annealing',
                                   bias=False, learning_rate=1, clip_max=1,
                                   max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        weights = np.array([1, 1, 1, 1])

        network.fit(X, y, init_weights=weights)
        fitted = network.fitted_weights

        assert (sum(fitted) < 4 and len(fitted) == 4 and min(fitted) >= -1
                and max(fitted) <= 1)

    @staticmethod
    def test_fit_genetic_alg():
        """Test fit method using the genetic_alg algorithm"""

        network = LinearRegression(algorithm='genetic_alg', bias=False,
                                   learning_rate=1, clip_max=1,
                                   max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        network.fit(X, y)
        fitted = network.fitted_weights

        assert (sum(fitted) < 4 and len(fitted) == 4 and min(fitted) >= -1
                and max(fitted) <= 1)

    @staticmethod
    def test_fit_gradient_descent():
        """Test fit method using the gradient_descent algorithm"""

        network = LinearRegression(algorithm='gradient_descent',
                                   bias=False, learning_rate=0.1,
                                   clip_max=1, max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        weights = np.array([1, 1, 1, 1])

        network.fit(X, y, init_weights=weights)
        fitted = network.fitted_weights

        assert (sum(fitted) <= 4 and len(fitted) == 4 and min(fitted) >= -1
                and max(fitted) <= 1)

    @staticmethod
    def test_predict_no_bias():
        """Test predict method with no bias term"""

        network = LinearRegression(algorithm='random_hill_climb', bias=False,
                                   learning_rate=1, clip_max=1,
                                   max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        network.fitted_weights = np.array([1, 1, 1, 1])
        network.node_list = [4, 1]
        network.output_activation = identity

        x = np.reshape(np.array([2, 0, 4, 4, 2, 1]), [6, 1])

        assert np.array_equal(network.predict(X), x)

    @staticmethod
    def test_predict_bias():
        """Test predict method with bias term"""

        network = LinearRegression(algorithm='random_hill_climb', bias=True,
                                   learning_rate=1, clip_max=1,
                                   max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        network.fitted_weights = np.array([1, 1, 1, 1, 1])
        network.node_list = [5, 1]
        network.output_activation = identity

        x = np.reshape(np.array([3, 1, 5, 5, 3, 2]), [6, 1])

        assert np.array_equal(network.predict(X), x)


class TestLogisticRegression(unittest.TestCase):
    """Tests for LogisticRegression class."""

    @staticmethod
    def test_fit_random_hill_climb():
        """Test fit method using the random hill climbing algorithm"""

        network = LogisticRegression(algorithm='random_hill_climb', bias=False,
                                     learning_rate=1, clip_max=1,
                                     max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        weights = np.array([1, 1, 1, 1])

        network.fit(X, y, init_weights=weights)
        fitted = network.fitted_weights

        assert (sum(fitted) < 4 and len(fitted) == 4 and min(fitted) >= -1
                and max(fitted) <= 1)

    @staticmethod
    def test_fit_simulated_annealing():
        """Test fit method using the simulated_annealing algorithm"""

        network = LogisticRegression(algorithm='simulated_annealing',
                                     bias=False, learning_rate=1, clip_max=1,
                                     max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        weights = np.array([1, 1, 1, 1])

        network.fit(X, y, init_weights=weights)
        fitted = network.fitted_weights

        assert (sum(fitted) < 4 and len(fitted) == 4 and min(fitted) >= -1
                and max(fitted) <= 1)

    @staticmethod
    def test_fit_genetic_alg():
        """Test fit method using the genetic_alg algorithm"""

        network = LogisticRegression(algorithm='genetic_alg', bias=False,
                                     learning_rate=1, clip_max=1,
                                     max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        network.fit(X, y)
        fitted = network.fitted_weights

        assert (sum(fitted) < 4 and len(fitted) == 4 and min(fitted) >= -1
                and max(fitted) <= 1)

    @staticmethod
    def test_fit_gradient_descent():
        """Test fit method using the gradient_descent algorithm"""

        network = LogisticRegression(algorithm='gradient_descent',
                                     bias=False, learning_rate=0.1, clip_max=1,
                                     max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])

        weights = np.array([1, 1, 1, 1])

        network.fit(X, y, init_weights=weights)
        fitted = network.fitted_weights

        assert (sum(fitted) <= 4 and len(fitted) == 4 and min(fitted) >= -1
                and max(fitted) <= 1)

    @staticmethod
    def test_predict_no_bias():
        """Test predict method with no bias term"""

        network = LogisticRegression(algorithm='random_hill_climb', bias=False,
                                     learning_rate=1, clip_max=1,
                                     max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        network.fitted_weights = np.array([-1, 1, 1, 1])
        network.node_list = [4, 1]
        network.output_activation = sigmoid

        probs = np.reshape(np.array([0.88080, 0.5, 0.88080, 0.88080, 0.88080,
                                     0.26894]), [6, 1])
        labels = np.reshape(np.array([1, 0, 1, 1, 1, 0]), [6, 1])

        assert (np.array_equal(network.predict(X), labels)
                and np.allclose(network.predicted_probs, probs, atol=0.0001))

    @staticmethod
    def test_predict_bias():
        """Test predict method with bias term"""

        network = LogisticRegression(algorithm='random_hill_climb', bias=True,
                                     learning_rate=1, clip_max=1,
                                     max_attempts=100)

        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])

        network.fitted_weights = np.array([-1, 1, 1, 1, -1])
        network.node_list = [5, 1]
        network.output_activation = sigmoid

        probs = np.reshape(np.array([0.73106, 0.26894, 0.73106, 0.73106,
                                     0.73106, 0.11920]), [6, 1])
        labels = np.reshape(np.array([1, 0, 1, 1, 1, 0]), [6, 1])

        assert (np.array_equal(network.predict(X), labels)
                and np.allclose(network.predicted_probs, probs, atol=0.0001))


if __name__ == '__main__':
    unittest.main()
