""" Unit tests for mlrose package

    Author: Genevieve Hayes
    License: 3-clause BSD license.
"""
import unittest
import numpy as np
from fitness import (OneMax, FlipFlop, head, tail, max_run, Queens,
                     MaxKColor)
from activation import identity, sigmoid, softmax, tanh, relu


class TestFitness(unittest.TestCase):
    """Tests for fitness.py."""

    @staticmethod
    def test_onemax():
        """Test OneMax fitness function"""
        state = np.array([0, 1, 0, 1, 1, 1, 1])
        assert OneMax().evaluate(state) == 5

    @staticmethod
    def test_flipflop():
        """Test FlipFlop fitness function"""
        state = np.array([0, 1, 0, 1, 1, 1, 1])
        assert FlipFlop().evaluate(state) == 3

    @staticmethod
    def test_head():
        """Test head function"""
        state = np.array([1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 4, 6, 1, 1])
        assert head(1, state) == 4

    @staticmethod
    def test_tail():
        """Test tail function"""
        state = np.array([1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 4, 6, 1, 1])
        assert tail(1, state) == 2

    @staticmethod
    def test_max_run():
        """Test max_run function"""
        state = np.array([1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 4, 6, 1, 1])
        assert max_run(1, state) == 5

    @staticmethod
    def test_queens():
        """Test Queens fitness function"""
        state = np.array([1, 4, 1, 3, 5, 5, 2, 7])
        assert Queens().evaluate(state) == 6

    @staticmethod
    def test_max_k_color():
        """Test MaxKColor fitness function"""

        edges = np.array([[0, 1, 1, 0, 1],
                          [1, 0, 0, 1, 0],
                          [1, 0, 0, 1, 0],
                          [0, 1, 1, 0, 1],
                          [1, 0, 0, 1, 0]])

        state = np.array([0, 1, 0, 1, 1])
        assert MaxKColor(edges).evaluate(state) == 3


class TestActivation(unittest.TestCase):
    """Tests for activation.py."""

    @staticmethod
    def test_identity():
        """Test identity activation function"""

        x = np.array([[0, 1, 3],
                      [-1, 0, -5],
                      [1, 0, 3],
                      [10, -9, -7]])

        assert np.array_equal(identity(x), x)

    @staticmethod
    def test_identity_deriv():
        """Test identity activation function derivative"""

        x = np.array([[0, 1, 3],
                      [-1, 0, -5],
                      [1, 0, 3],
                      [10, -9, -7]])

        assert np.array_equal(identity(x, deriv=True), np.ones([4, 3]))

    @staticmethod
    def test_sigmoid():
        """Test sigmoid activation function"""

        x = np.array([[0, 1, 3],
                      [-1, 0, -5],
                      [1, 0, 3],
                      [10, -9, -7]])

        y = np.array([[0.5, 0.73106, 0.95257],
                      [0.26894, 0.5, 0.00669],
                      [0.73106, 0.5, 0.95257],
                      [0.99995, 0.00012, 0.00091]])

        assert np.allclose(sigmoid(x), y, atol=0.00001)

    @staticmethod
    def test_sigmoid_deriv():
        """Test sigmoid activation function derivative"""

        x = np.array([[0, 1, 3],
                      [-1, 0, -5],
                      [1, 0, 3],
                      [10, -9, -7]])

        y = np.array([[0.5, 0.73106, 0.95257],
                      [0.26894, 0.5, 0.00669],
                      [0.73106, 0.5, 0.95257],
                      [0.99995, 0.00012, 0.00091]])

        assert np.allclose(sigmoid(x, deriv=True), y*(1-y), atol=0.00001)

    @staticmethod
    def test_relu():
        """Test relu activation function"""

        x = np.array([[0, 1, 3],
                      [-1, 0, -5],
                      [1, 0, 3],
                      [10, -9, -7]])

        y = np.array([[0, 1, 3],
                      [0, 0, 0],
                      [1, 0, 3],
                      [10, 0, 0]])

        assert np.array_equal(relu(x), y)

    @staticmethod
    def test_relu_deriv():
        """Test relu activation function derivative"""

        x = np.array([[0, 1, 3],
                      [-1, 0, -5],
                      [1, 0, 3],
                      [10, -9, -7]])

        y = np.array([[0, 1, 1],
                      [0, 0, 0],
                      [1, 0, 1],
                      [1, 0, 0]])

        assert np.array_equal(relu(x, deriv=True), y)

    @staticmethod
    def test_tanh():
        """Test tanh activation function"""

        x = np.array([[0, 1, 3],
                      [-1, 0, -5],
                      [1, 0, 3],
                      [10, -9, -7]])

        y = np.array([[0, 0.76159, 0.99505],
                      [-0.76159, 0, -0.99991],
                      [0.76159, 0, 0.99505],
                      [1.00000, -1.00000, -1.00000]])

        assert np.allclose(tanh(x), y, atol=0.00001)

    @staticmethod
    def test_tanh_deriv():
        """Test tanh activation function derivative"""

        x = np.array([[0, 1, 3],
                      [-1, 0, -5],
                      [1, 0, 3],
                      [10, -9, -7]])

        y = np.array([[0, 0.76159, 0.99505],
                      [-0.76159, 0, -0.99991],
                      [0.76159, 0, 0.99505],
                      [1.00000, -1.00000, -1.00000]])

        assert np.allclose(tanh(x, deriv=True), 1-y**2, atol=0.00001)

    @staticmethod
    def test_softmax():
        """Test softmax activation function"""

        x = np.array([[0, 1, 3],
                      [-1, 0, -5],
                      [1, 0, 3],
                      [10, -9, -7]])

        y = np.array([[1, 2.71828, 20.08554],
                      [0.36788, 1, 0.00674],
                      [2.71828, 1, 20.08554],
                      [22026.46579, 0.00012, 0.00091]])

        assert np.allclose(softmax(x), y/np.reshape(np.sum(y, axis=1), [4, 1]),
                           atol=0.00001)


if __name__ == '__main__':
    unittest.main()
