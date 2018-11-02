""" Unit tests for mlrose package

    Author: Genevieve Hayes <ghayes17@gmail.com>
    License: 3-clause BSD license.
"""
import unittest
import numpy as np
from fitness import (OneMax, FlipFlop, head, tail, max_run, Queens, 
                     MaxKColor)


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


if __name__ == '__main__':
    unittest.main()
