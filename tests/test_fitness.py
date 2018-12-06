""" Unit tests for fitness.py"""

# Author: Genevieve Hayes
# License: BSD 3 clause

import unittest
import numpy as np
from mlrose import (OneMax, FlipFlop, FourPeaks, SixPeaks, ContinuousPeaks,
                    Knapsack, TravellingSales, Queens, MaxKColor,
                    CustomFitness)
from mlrose.fitness import head, tail, max_run
# The above functions are not automatically imported at initialization, so
# must be imported explicitly from fitness.py.


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
    def test_travelling_sales_coords():
        """Test TravellingSales fitness function for case where city nodes
        coords are specified."""

        coords = [(0, 0), (3, 0), (3, 2), (2, 4), (1, 3)]

        state = np.array([0, 1, 4, 3, 2])

        assert (round(TravellingSales(coords=coords).evaluate(state), 4)
                == 13.8614)

    @staticmethod
    def test_travelling_sales_dists():
        """Test TravellingSales fitness function for case where distances
        between node pairs are specified."""

        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6),
                 (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]

        state = np.array([0, 1, 4, 3, 2])

        assert TravellingSales(distances=dists).evaluate(state) == 29

    @staticmethod
    def test_travelling_sales_invalid():
        """Test TravellingSales fitness function for invalid tour"""

        dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6),
                 (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]

        state = np.array([0, 1, 2, 3, 4])

        assert TravellingSales(distances=dists).evaluate(state) == np.inf

    @staticmethod
    def test_queens():
        """Test Queens fitness function"""
        state = np.array([1, 4, 1, 3, 5, 5, 2, 7])
        assert Queens().evaluate(state) == 6

    @staticmethod
    def test_max_k_color():
        """Test MaxKColor fitness function"""
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]

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
