""" Unit tests for opt_probs.py

    Author: Genevieve Hayes
    License: 3-clause BSD license.
"""
import unittest
import numpy as np
from fitness import OneMax
from opt_probs import OptProb, DiscreteOpt, ContinuousOpt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_tree

class TestOptProb(unittest.TestCase):
    """Tests for OptProb class."""

    @staticmethod
    def test_set_state_max():
        """Test set_state method for a maximization problem"""
        
        problem = OptProb(5, OneMax(), maximize = True)
        
        x = np.array([0, 1, 2, 3, 4])
        problem.set_state(x)
        
        assert np.array_equal(problem.get_state(), x) \
               and problem.get_fitness() == 10
               
    @staticmethod
    def test_set_state_min():
        """Test set_state method for a minimization problem"""
        
        problem = OptProb(5, OneMax(), maximize = False)
        
        x = np.array([0, 1, 2, 3, 4])
        problem.set_state(x)
        
        assert np.array_equal(problem.get_state(), x) \
               and problem.get_fitness() == -10
    
    @staticmethod
    def test_set_population_max():
        """Test set_population method for a maximization problem"""
        
        problem = OptProb(5, OneMax(), maximize = True)
        
        pop = np.array([[0, 0, 0, 0, 1], 
                        [1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [100, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, -50]])
    
        pop_fit = np.array([1, 3, 4, 2, 100, 0, 5, -50])
    
        problem.set_population(pop)
        
        assert np.array_equal(problem.get_population(), pop) \
               and np.array_equal(problem.get_pop_fitness(), pop_fit)
               
    @staticmethod
    def test_set_population_min():
        """Test set_population method for a minimization problem"""
        
        problem = OptProb(5, OneMax(), maximize = False)
        
        pop = np.array([[0, 0, 0, 0, 1], 
                        [1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [100, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, -50]])
    
        pop_fit = -1.0*np.array([1, 3, 4, 2, 100, 0, 5, -50])
    
        problem.set_population(pop)
        
        assert np.array_equal(problem.get_population(), pop) \
               and np.array_equal(problem.get_pop_fitness(), pop_fit)
               
    @staticmethod
    def test_best_child_max():
        """Test best_child method for a maximization problem"""

        problem = OptProb(5, OneMax(), maximize = True)
        
        pop = np.array([[0, 0, 0, 0, 1], 
                        [1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [100, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, -50]])
        
        problem.set_population(pop)
        x = problem.best_child()

        assert np.array_equal(x, np.array([100, 0, 0, 0, 0]))

    @staticmethod
    def test_best_child_min():
        """Test best_child method for a minimization problem"""

        problem = OptProb(5, OneMax(), maximize = False)
        
        pop = np.array([[0, 0, 0, 0, 1], 
                        [1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [100, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, -50]])
        
        problem.set_population(pop)
        x = problem.best_child()

        assert np.array_equal(x, np.array([0, 0, 0, 0, -50]))
        
    @staticmethod
    def test_best_neighbor_max():
        """Test best_neighbor method for a maximization problem"""

        problem = OptProb(5, OneMax(), maximize = True)
        
        pop = np.array([[0, 0, 0, 0, 1], 
                        [1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [100, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, -50]])
        
        problem.neighbors = pop
        x = problem.best_neighbor()

        assert np.array_equal(x, np.array([100, 0, 0, 0, 0]))

    @staticmethod
    def test_best_neighbor_min():
        """Test best_neighbor method for a minimization problem"""

        problem = OptProb(5, OneMax(), maximize = False)
        
        pop = np.array([[0, 0, 0, 0, 1], 
                        [1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [100, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, -50]])
        
        problem.neighbors = pop
        x = problem.best_neighbor()

        assert np.array_equal(x, np.array([0, 0, 0, 0, -50]))
    
    @staticmethod
    def test_eval_fitness_max():
        """Test eval_fitness method for a maximization problem"""
        
        problem = OptProb(5, OneMax(), maximize = True)
        x = np.array([0, 1, 2, 3, 4])
        fitness = problem.eval_fitness(x)
        
        assert fitness == 10
    
    @staticmethod
    def test_eval_fitness_min():
        """Test eval_fitness method for a minimization problem"""
        
        problem = OptProb(5, OneMax(), maximize = False)
        x = np.array([0, 1, 2, 3, 4])
        fitness = problem.eval_fitness(x)
        
        assert fitness == -10
    
    @staticmethod
    def test_eval_mate_probs():
        """Test eval_mate_probs method"""
        
        problem = OptProb(5, OneMax(), maximize = True)
        pop = np.array([[0, 0, 0, 0, 1], 
                        [1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1]])
       
        problem.set_population(pop)
        problem.eval_mate_probs()
        
        probs = np.array([0.06667, 0.2, 0.26667, 0.13333, 0, 0.33333])
        
        assert np.allclose(problem.get_mate_probs(), probs, atol=0.00001)
        
    @staticmethod
    def test_eval_mate_probs_all_zero():
        """Test eval_mate_probs method when all states have zero fitness"""
        
        problem = OptProb(5, OneMax(), maximize = True)
        pop = np.array([[0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]])
       
        problem.set_population(pop)
        problem.eval_mate_probs()
        
        probs = np.array([0.16667, 0.16667, 0.16667, 0.16667, 
                          0.16667, 0.16667])

        assert np.allclose(problem.get_mate_probs(), probs, atol=0.00001)
       
class TestDiscreteOpt(unittest.TestCase):
    """Tests for DiscreteOpt class."""
    
    @staticmethod
    def test_eval_node_probs():
        """Test eval_node_probs method"""
        
        problem = DiscreteOpt(5, OneMax(), maximize = True)
        
        pop = np.array([[0, 0, 0, 0, 1], 
                        [1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1]])
        
        problem.keep_sample = pop
        problem.eval_node_probs()

        parent = np.array([2, 0, 1, 0])
        probs = np.array([[[0.33333, 0.66667],
                           [0.33333, 0.66667]],
                          [[1.0, 0.0],
                           [0.33333, 0.66667]],
                          [[1.0, 0.0],
                           [0.25, 0.75]],
                          [[1.0, 0.0],
                           [0.0, 1.0]],
                          [[0.5, 0.5],
                           [0.25, 0.75]]])
        
        assert np.allclose(problem.node_probs, probs, atol=0.00001) \
               and np.array_equal(problem.parent_nodes, parent)
    
    @staticmethod
    def test_find_neighbors_max2():
        """Test find_neighbors method when max_val is equal to 2"""
        
        problem = DiscreteOpt(5, OneMax(), maximize = True, max_val = 2)
        
        x = np.array([0, 1, 0, 1, 0])
        problem.set_state(x)
        problem.find_neighbors()
        
        neigh = np.array([[1, 1, 0, 1, 0],
                          [0, 0, 0, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 0, 0, 0],
                          [0, 1, 0, 1, 1]])
        
        assert np.array_equal(np.array(problem.neighbors), neigh)
                
    @staticmethod
    def test_find_neighbors_max_gt2():
        """Test find_neighbors method when max_val is greater than 2"""
        
        problem = DiscreteOpt(5, OneMax(), maximize = True, max_val = 3)
        
        x = np.array([0, 1, 2, 1, 0])
        problem.set_state(x)
        problem.find_neighbors()
        
        neigh = np.array([[1, 1, 2, 1, 0],
                          [2, 1, 2, 1, 0],
                          [0, 0, 2, 1, 0],
                          [0, 2, 2, 1, 0],
                          [0, 1, 0, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 2, 0, 0],
                          [0, 1, 2, 2, 0],
                          [0, 1, 2, 1, 1],
                          [0, 1, 2, 1, 2]])
        
        assert np.array_equal(np.array(problem.neighbors), neigh)
    
    @staticmethod
    def test_find_sample_order():
        """Test find_sample_order method"""
        
        problem = DiscreteOpt(5, OneMax(), maximize = True)
        problem.parent_nodes = np.array([2, 0, 1, 0])
        
        order = np.array([0, 2, 4, 1, 3])
        
        assert np.array_equal(np.array(problem.find_sample_order()), order)
        
    @staticmethod
    def test_find_top_pct_max():
        """Test find_top_pct method for a maximization problem"""
        
        problem = DiscreteOpt(5, OneMax(), maximize = True)
        
        pop = np.array([[0, 0, 0, 0, 1], 
                        [1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [100, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, -50]])
        
        problem.set_population(pop)
        problem.find_top_pct(keep_pct = 0.25)
        
        x = np.array([[100, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1]])
    
        assert np.array_equal(problem.get_keep_sample(), x)
    
    @staticmethod
    def test_find_top_pct_min():
        """Test find_top_pct method for a minimization problem"""
        
        problem = DiscreteOpt(5, OneMax(), maximize = False)
        
        pop = np.array([[0, 0, 0, 0, 1], 
                        [1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [100, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 0, -50]])
        
        problem.set_population(pop)
        problem.find_top_pct(keep_pct = 0.25)
        
        x = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, -50]])
    
        assert np.array_equal(problem.get_keep_sample(), x)
        
    @staticmethod
    def test_random():
        """Test random method"""
        
        problem = DiscreteOpt(5, OneMax(), maximize = True, max_val = 5)
        
        rand = problem.random()
        
        assert len(rand) == 5 and max(rand) >= 0 and min(rand) <= 4   
    
    @staticmethod
    def test_random_neighbor_max2():
        """Test random_neighbor method when max_val is equal to 2"""
        
        problem = DiscreteOpt(5, OneMax(), maximize = True)
        
        x = np.array([0, 0, 1, 1, 1])
        problem.set_state(x)
        
        neigh = problem.random_neighbor()
        sum_diff = np.sum(np.abs(x - neigh))
        
        assert len(neigh) == 5 and sum_diff == 1
    
    @staticmethod
    def test_random_neighbor_max_gt2():
        """Test random_neighbor method when max_val is greater than 2"""
        
        problem = DiscreteOpt(5, OneMax(), maximize = True, max_val = 5)
        
        x = np.array([0, 1, 2, 3, 4])
        problem.set_state(x)
        
        neigh = problem.random_neighbor()
        abs_diff = np.abs(x - neigh)
        abs_diff[abs_diff > 0] = 1

        sum_diff = np.sum(abs_diff)
        
        assert len(neigh) == 5 and sum_diff == 1
    
    @staticmethod
    def test_random_pop():
        """Test random_pop method"""
        
        problem = DiscreteOpt(5, OneMax(), maximize = True)
        problem.random_pop(100)
        
        pop = problem.get_population()
        pop_fitness = problem.get_pop_fitness()
        
        assert np.shape(pop)[0] == 100 and np.shape(pop)[1] == 5 \
               and np.sum(pop) > 0 and np.sum(pop) < 500 \
               and len(pop_fitness) == 100
               
    @staticmethod
    def test_reproduce_mut0():
        """Test reproduce method when mutation_prob is 0"""
        
        problem = DiscreteOpt(5, OneMax(), maximize = True)
        father = np.array([0, 0, 0, 0, 0])
        mother = np.array([1, 1, 1, 1, 1])
        
        child = problem.reproduce(father, mother, mutation_prob = 0)

        assert len(child) == 5 and sum(child) > 0 and sum(child) < 5
    
    @staticmethod
    def test_reproduce_mut1_max2():
        """Test reproduce method when mutation_prob is 1 and max_val is 2"""
        
        problem = DiscreteOpt(5, OneMax(), maximize = True)
        father = np.array([0, 0, 0, 0, 0])
        mother = np.array([1, 1, 1, 1, 1])
        
        child = problem.reproduce(father, mother, mutation_prob = 1)

        assert len(child) == 5 and sum(child) > 0 and sum(child) < 5
    
    @staticmethod
    def test_reproduce_mut1_max_gt2():
        """Test reproduce method when mutation_prob is 1 and max_val is 
        greater than 2"""
        
        problem = DiscreteOpt(5, OneMax(), maximize = True, max_val = 3)
        father = np.array([0, 0, 0, 0, 0])
        mother = np.array([2, 2, 2, 2, 2])
        
        child = problem.reproduce(father, mother, mutation_prob = 1)

        assert len(child) == 5 and sum(child) > 0 and sum(child) < 10
    
    @staticmethod
    def test_sample_pop():
        """Test sample_pop method"""
        
        problem = DiscreteOpt(5, OneMax(), maximize = True)
        
        pop = np.array([[0, 0, 0, 0, 1], 
                        [1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1]])
        
        problem.keep_sample = pop
        problem.eval_node_probs()
        
        sample = problem.sample_pop(100)
        
        assert np.shape(sample)[0] == 100 and np.shape(sample)[1] == 5 \
               and np.sum(sample) > 0 and np.sum(sample) < 500
               
class TestContinuousOpt(unittest.TestCase):
    """Tests for ContinuousOpt class."""
    
    @staticmethod
    def test_find_neighbors_range_eq_step():
        """Test find_neighbors method when range equals step size"""
        
        problem = ContinuousOpt(5, OneMax(), maximize = True, 
                                min_val = 0, max_val = 1, step = 1)
    
        x = np.array([0, 1, 0, 1, 0])
        problem.set_state(x)
        
        problem.find_neighbors()
        
        neigh = np.array([[1, 1, 0, 1, 0],
                          [0, 0, 0, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 0, 0, 0],
                          [0, 1, 0, 1, 1]])
        
        assert np.array_equal(np.array(problem.neighbors), neigh)
        
    @staticmethod
    def test_find_neighbors_range_gt_step():
        """Test find_neighbors method when range greater than step size"""
        
        problem = ContinuousOpt(5, OneMax(), maximize = True, 
                                min_val = 0, max_val = 2, step = 1)
        
        x = np.array([0, 1, 2, 1, 0])
        problem.set_state(x)
        problem.find_neighbors()
        
        neigh = np.array([[1, 1, 2, 1, 0],
                          [0, 0, 2, 1, 0],
                          [0, 2, 2, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 2, 0, 0],
                          [0, 1, 2, 2, 0],
                          [0, 1, 2, 1, 1]])
        
        assert np.array_equal(np.array(problem.neighbors), neigh)
    
    @staticmethod
    def test_random():
        """Test random method"""
        
        problem = ContinuousOpt(5, OneMax(), maximize = True, 
                                min_val = 0, max_val = 4)
        
        rand = problem.random()
        
        assert len(rand) == 5 and max(rand) >= 0 and min(rand) <= 4
    
    @staticmethod
    def test_random_neighbor_range_eq_step():
        """Test random_neighbor method when range equals step size"""
        
        problem = ContinuousOpt(5, OneMax(), maximize = True, 
                                min_val = 0, max_val = 1, step = 1)
        
        x = np.array([0, 0, 1, 1, 1])
        problem.set_state(x)
        
        neigh = problem.random_neighbor()
        sum_diff = np.sum(np.abs(x - neigh))
        
        assert len(neigh) == 5 and sum_diff == 1
    
    @staticmethod
    def test_random_neighbor_range_gt_step():
        """Test random_neighbor method when range greater than step size"""
        
        problem = ContinuousOpt(5, OneMax(), maximize = True, 
                                min_val = 0, max_val = 2, step = 1)
        
        x = np.array([0, 1, 2, 3, 4])
        problem.set_state(x)
        
        neigh = problem.random_neighbor()
        abs_diff = np.abs(x - neigh)
        abs_diff[abs_diff > 0] = 1

        sum_diff = np.sum(abs_diff)
        
        assert len(neigh) == 5 and sum_diff == 1
        
    @staticmethod
    def test_random_pop():
        """Test random_pop method"""
        
        problem = ContinuousOpt(5, OneMax(), maximize = True, 
                                min_val = 0, max_val = 1, step = 1)
        problem.random_pop(100)
        
        pop = problem.get_population()
        pop_fitness = problem.get_pop_fitness()
        
        assert np.shape(pop)[0] == 100 and np.shape(pop)[1] == 5 \
               and np.sum(pop) > 0 and np.sum(pop) < 500 \
               and len(pop_fitness) == 100
    
    @staticmethod
    def test_reproduce_mut0():
        """Test reproduce method when mutation_prob is 0"""
        
        problem = ContinuousOpt(5, OneMax(), maximize = True, 
                                min_val = 0, max_val = 1, step = 1)
        father = np.array([0, 0, 0, 0, 0])
        mother = np.array([1, 1, 1, 1, 1])
        
        child = problem.reproduce(father, mother, mutation_prob = 0)

        assert len(child) == 5 and sum(child) > 0 and sum(child) < 5
    
    @staticmethod
    def test_reproduce_mut1_range_eq_step():
        """Test reproduce method when mutation_prob is 1 and range equals 
        step size"""
        
        problem = ContinuousOpt(5, OneMax(), maximize = True, 
                                min_val = 0, max_val = 1, step = 1)
        father = np.array([0, 0, 0, 0, 0])
        mother = np.array([1, 1, 1, 1, 1])
        
        child = problem.reproduce(father, mother, mutation_prob = 1)

        assert len(child) == 5 and sum(child) > 0 and sum(child) < 5
    
    @staticmethod
    def test_reproduce_mut1_range_gt_step():
        """Test reproduce method when mutation_prob is 1 and range is 
        greater than step size"""
        
        problem = ContinuousOpt(5, OneMax(), maximize = True, 
                                min_val = 0, max_val = 2, step = 1)
        father = np.array([0, 0, 0, 0, 0])
        mother = np.array([2, 2, 2, 2, 2])
        
        child = problem.reproduce(father, mother, mutation_prob = 1)

        assert len(child) == 5 and sum(child) > 0 and sum(child) < 10
    

if __name__ == '__main__':
    unittest.main()

    