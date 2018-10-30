""" Unit tests for mlrose package

    Author: Genevieve Hayes <ghayes17@gmail.com>
    License: 3-clause BSD license.
"""

import unittest
from fitness import *
'''
from algorithms import *
from discrete import *
from fitness import *
from decay import *
from neural import *
import numpy as np
'''

class TestFitness(unittest.TestCase):
    """Tests for fitness.py."""

    def test_onemax(self):
        """Test OneMax fitness function"""        
        state = np.array([0, 1, 0, 1, 1, 1, 1])
        assert(OneMax().evaluate(state) == 5)
        
    def test_flipflop(self):
        """Test FlipFlop fitness function"""        
        state = np.array([0, 1, 0, 1, 1, 1, 1])
        assert(FlipFlop().evaluate(state) == 3)
        
    def test_head(self):
        """Test head function"""
        state = np.array([1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 4, 6, 1, 1])
        assert(head(1, state) == 4)
    
    def test_tail(self):
        """Test tail function"""
        state = np.array([1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 4, 6, 1, 1])
        assert(tail(1, state) == 2)
    
    def test_max_run(self):
        """Test max_run function"""
        state = np.array([1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 4, 6, 1, 1])
        assert(max_run(1, state) == 5)
   
    def test_queens(self):
        """Test Queens fitness function"""
        state = np.array([1, 4, 1, 3, 5, 5, 2, 7])
        assert(Queens().evaluate(state) == 6)
        
    
if __name__ == '__main__':
    unittest.main()