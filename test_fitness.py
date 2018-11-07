""" Unit tests for fitness.py

    Author: Genevieve Hayes
    License: 3-clause BSD license.
"""
import unittest
import numpy as np
from fitness import (OneMax, FlipFlop, head, tail, max_run, FourPeaks,
                     SixPeaks, ContinuousPeaks, Knapsack, TravellingSales,
                     Queens, MaxKColor, CustomFitness)


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
    def test_max_run_middle():
        """Test max_run function for case where run is in the middle of the
        state"""
        state = np.array([1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 4, 6, 1, 1])
        assert max_run(1, state) == 5

    @staticmethod
    def test_max_run_start():
        """Test max_run function for case where run is at the start of the
        state"""
        state = np.array([1, 1, 1, 1, 1, 1, 0, 2, 1, 1, 1, 1, 1, 4, 6, 1, 1])
        assert max_run(1, state) == 6

    @staticmethod
    def test_max_run_end():
        """Test max_run function for case where run is at the end of the
        state"""
        state = np.array([1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert max_run(1, state) == 9

    @staticmethod
    def test_fourpeaks_r0():
        """Test FourPeaks fitness function for the case where R=0 and max>0"""
        state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        assert FourPeaks(t_pct=0.30).evaluate(state) == 4

    @staticmethod
    def test_fourpeaks_r_gt0():
        """Test FourPeaks fitness function for the case where R>0 and max>0"""
        state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        assert FourPeaks(t_pct=0.15).evaluate(state) == 16

    @staticmethod
    def test_fourpeaks_r0_max0():
        """Test FourPeaks fitness function for the case where R=0 and max=0"""
        state = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1])
        assert FourPeaks(t_pct=0.30).evaluate(state) == 0

    @staticmethod
    def test_sixpeaks_r0():
        """Test SixPeaks fitness function for the case where R=0 and max>0"""
        state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        assert SixPeaks(t_pct=0.30).evaluate(state) == 4

    @staticmethod
    def test_sixpeaks_r_gt0():
        """Test SixPeaks fitness function for the case where R>0 and max>0"""
        state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        assert SixPeaks(t_pct=0.15).evaluate(state) == 16

    @staticmethod
    def test_sixpeaks_r0_max0():
        """Test SixPeaks fitness function for the case where R=0 and max=0"""
        state = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1])
        assert SixPeaks(t_pct=0.30).evaluate(state) == 0

    @staticmethod
    def test_sixpeaks_r_gt0_max0():
        """Test SixPeaks fitness function for the case where R>0 and max=0"""
        state = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1])
        assert SixPeaks(t_pct=0.15).evaluate(state) == 12

    @staticmethod
    def test_continuouspeaks_r0():
        """Test ContinuousPeaks fitness function for case when R = 0."""
        state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1])
        assert ContinuousPeaks(t_pct=0.30).evaluate(state) == 5

    @staticmethod
    def test_continuouspeaks_r_gt():
        """Test ContinuousPeaks fitness function for case when R > 0."""
        state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1])

        assert ContinuousPeaks(t_pct=0.15).evaluate(state) == 17

    @staticmethod
    def test_knapsack_weight_lt_max():
        """Test Knapsack fitness function for case where total weight is less
        than the maximum"""
        weights = [10, 5, 2, 8, 15]
        values = [1, 2, 3, 4, 5]
        max_weight_pct = 0.6

        state = np.array([1, 0, 2, 1, 0])
        assert Knapsack(weights, values, max_weight_pct).evaluate(state) == 11

    @staticmethod
    def test_knapsack_weight_gt_max():
        """Test Knapsack fitness function for case where total weight is
        greater than the maximum"""
        weights = [10, 5, 2, 8, 15]
        values = [1, 2, 3, 4, 5]
        max_weight_pct = 0.4

        state = np.array([1, 0, 2, 1, 0])
        assert Knapsack(weights, values, max_weight_pct).evaluate(state) == 0

    @staticmethod
    def test_travelling_sales_nodups():
        """Test TravellingSales fitness function for case where each city is
        visited only once"""
        distance = np.array([[0, 3, 5, 1, 7],
                             [3, 0, -1, 6, 9],
                             [5, -1, 0, 8, 2],
                             [1, 6, 8, 0, 4],
                             [7, 9, 2, 4, 0]])

        state = np.array([0, 1, 0, 0, 0,
                          0, 0, 0, 0, 1,
                          1, 0, 0, 0, 0,
                          0, 0, 1, 0, 0,
                          0, 0, 0, 1, 0])

        assert TravellingSales(distance).evaluate(state) == 29

    @staticmethod
    def test_travelling_sales_dups():
        """Test TravellingSales fitness function for invalid tour"""
        distance = np.array([[0, 3, 5, 1, 7],
                             [3, 0, -1, 6, 9],
                             [5, -1, 0, 8, 2],
                             [1, 6, 8, 0, 4],
                             [7, 9, 2, 4, 0]])

        state = np.array([0, 1, 0, 0, 1,
                          0, 0, 0, 0, 1,
                          1, 0, 1, 0, 0,
                          0, 0, 1, 0, 0,
                          0, 1, 0, 1, 0])

        assert TravellingSales(distance).evaluate(state) == 0

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

    @staticmethod
    def test_custom_fitness():
        """Test CustomFitness fitness function"""
        # Define custom finess function
        def cust_fn(state, c):
            return c*np.sum(state)

        state = np.array([1, 2, 3, 4, 5])
        kwargs = {'c': 10}
        assert CustomFitness(cust_fn, **kwargs).evaluate(state) == 150


if __name__ == '__main__':
    unittest.main()
