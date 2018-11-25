""" Unit tests for neural.py

    Author: Genevieve Hayes
    License: 3-clause BSD license.
"""
import unittest
import numpy as np
from neural import (flatten_weights, unflatten_weights, gradient_descent,
                    NetworkWeights, NeuralNetwork)
from activation import identity

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

        assert np.array_equal(weights[0], a) \
               and np.array_equal(weights[1], b) \
               and np.array_equal(weights[2], c)
    

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
        
        fitness = NetworkWeights(X, y, nodes, activation = identity, 
                                 bias = False)
        
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
        
        fitness = NetworkWeights(X, y, nodes, activation = identity, 
                                 bias = False)
        
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
        
        fitness = NetworkWeights(X, y, nodes, activation = identity, 
                                 bias = False, is_classifier = False)
        
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
        
        fitness = NetworkWeights(X, y, nodes, activation = identity, 
                                 bias = True, is_classifier = False)
        
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
        
        fitness = NetworkWeights(X, y, nodes, activation = identity, 
                                 bias = False, is_classifier = False,
                                 learning_rate = 1)
        
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
        
        assert  np.allclose(updates[0], update1, atol=0.001) \
                and np.allclose(updates[1], update2, atol=0.001)
   
    
class TestNeuralNetwork(unittest.TestCase):
    """Tests for NeuralNetwork class."""
    
    @staticmethod
    def test_fit_random_hill_climb_classifier():
        """Test fit method for a classifier using the random hill climbing 
        algorithm"""
        
        network = NeuralNetwork(hidden_nodes = [2], activation = 'identity',
                                algorithm = 'random_hill_climb', max_iters = 1, 
                                bias = False, is_classifier=True, 
                                learning_rate = 1, max_attempts=100)
        
        X = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0]])
        
        y = np.reshape(np.array([1, 1, 0, 0, 1, 1]), [6, 1])
        
        weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        
        network.fit(X, y, init_weights=weights)

        assert abs(sum(network.fitted_weights)) == 9

    
if __name__ == '__main__':
    unittest.main()