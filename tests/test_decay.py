""" Unit tests for decay.py"""

# Author: Genevieve Hayes
# License: BSD 3 clause

try:
    import mlrose_hiive
except:
    import sys
    sys.path.append("..")
import unittest
from mlrose_hiive import GeomDecay, ArithDecay, ExpDecay, CustomSchedule


class TestDecay(unittest.TestCase):
    """Tests for decay.py."""

    @staticmethod
    def test_geom_above_min():
        """Test geometric decay evaluation function for case where result is
        above the minimum"""

        schedule = GeomDecay(init_temp=10, decay=0.95, min_temp=1)
        x = schedule.evaluate(5)

        assert round(x, 5) == 7.73781

    @staticmethod
    def test_geom_below_min():
        """Test geometric decay evaluation function for case where result is
        below the minimum"""

        schedule = GeomDecay(init_temp=10, decay=0.95, min_temp=1)
        x = schedule.evaluate(50)

        assert x == 1

    @staticmethod
    def test_arith_above_min():
        """Test arithmetic decay evaluation function for case where result is
        above the minimum"""

        schedule = ArithDecay(init_temp=10, decay=0.95, min_temp=1)
        x = schedule.evaluate(5)

        assert x == 5.25

    @staticmethod
    def test_arith_below_min():
        """Test arithmetic decay evaluation function for case where result is
        below the minimum"""

        schedule = ArithDecay(init_temp=10, decay=0.95, min_temp=1)
        x = schedule.evaluate(50)

        assert x == 1

    @staticmethod
    def test_exp_above_min():
        """Test exponential decay evaluation function for case where result is
        above the minimum"""

        schedule = ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)
        x = schedule.evaluate(5)

        assert round(x, 5) == 7.78801

    @staticmethod
    def test_exp_below_min():
        """Test exponential decay evaluation function for case where result is
        below the minimum"""

        schedule = ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)
        x = schedule.evaluate(50)

        assert x == 1

    @staticmethod
    def test_custom():
        """Test custom evaluation function"""
        # Define custom schedule function
        def custom(t, c):
            return t + c

        kwargs = {'c': 10}

        schedule = CustomSchedule(custom, **kwargs)
        x = schedule.evaluate(5)

        assert x == 15


if __name__ == '__main__':
    unittest.main()
