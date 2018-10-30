""" Classes for defining fitness functions.

    Author: Genevieve Hayes <ghayes17@gmail.com>
    License: 3-clause BSD license.
"""
import numpy as np

class OneMax:
    """Fitness function for One Max optimization problem."""
        
    def evaluate(self, state):
        """Evaluate the fitness of a state vector
        
        Args:
        state: array. State array for evaluation.
           
        Returns:
        fitness: float. Value of fitness function. 
        """
        
        fitness = sum(state)
        return fitness


class FlipFlop:
    """Fitness function for Flip Flop optimization problem."""
        
    def evaluate(self, state):
        """Evaluate the fitness of a state vector
        
        Args:
        state: array. State array for evaluation.
           
        Returns:
        fitness: float. Value of fitness function. 
        """
        fitness = 0
        
        for i in range(1, len(state)):
            if state[i] != state[i - 1]:
                fitness += 1
        
        return fitness
    
    
def head(b, x):
    """Determine the number of leading b's in vector x
    
    Args: 
    b: int. Integer for counting at head of vector.
    x: array. Vector of integers.
    
    Returns:
    head: int. Number of leading b's in x.
    """
    
    # Initialize counter
    head = 0
    
    # Iterate through values in vector
    for i in range(len(x)):
        if x[i] == b:
            head += 1
        else:
            break

    return head

def tail(b, x):
    """Determine the number of trailing b's in vector x
    
    Args: 
    b: int. Integer for counting at tail of vector.
    x: array. Vector of integers.
    
    Returns:
    tail: int. Number of trailing b's in x.
    """
    
    # Initialize counter
    tail = 0
    
    # Iterate backwards through values in vector
    for i in range(len(x)):
        if x[len(x) - i - 1] == b:
            tail += 1
        else:
            break

    return tail

def max_run(b, x):
    """Determine the length of the maximum run of b's in vector x
    
    Args: 
    b: int. Integer for counting.
    x: array. Vector of integers.
    
    Returns:
    max_run: int. Length of maximum run of b's
    """
    # Initialize counter
    max_run = 0
    run = 0
    
    # Iterate through values in vector
    for i in range(len(x)):
        if x[i] == b:
            run += 1
        else:
            if run > max_run:
                max_run = run
                
            run = 0

    return max_run
    

class FourPeaks:
    """Fitness function for Four Peaks optimization problem."""
    
    def __init__(self, t_pct = 0.1):
        """Initialize FourPeaks object.
        
        Args:
        t_pct: float. Threshold parameter (T) for four peaks fitness function.
        
        Returns:
        None            
        """
        self.t_pct = t_pct
        
    def evaluate(self, state):
        """Evaluate the fitness of a state
        
        Args:
        state: array. State array for evaluation.
           
        Returns:
        fitness: float. Value of fitness function. 
        """
        n = len(state)
        t = np.ceil(self.t_pct*n)
        
        # Calculate head and tail values
        tail_0 = tail(0, state)
        head_1 = head(1, state)

        # Calculate R(X, T)
        if (tail_0 > t and head_1 > t):
            r = n
        else:
            r = 0
        
        # Evaluate function
        fitness = max(tail_0, head_1) + r
    
        return fitness
    
    
class SixPeaks:
    """Fitness function for Six Peaks optimization problem."""
    
    def __init__(self, t_pct = 0.1):
        """Initialize SixPeaks object.
        
        Args:
        t_pct: float. Threshold parameter (T) for six peaks fitness function.
        
        Returns:
        None            
        """
        self.t_pct = t_pct
        
    def evaluate(self, state):
        """Evaluate the fitness of a state
        
        Args:
        state: array. State array for evaluation.
           
        Returns:
        fitness: float. Value of fitness function. 
        """
        n = len(state)
        t = np.ceil(self.t_pct*n)
        
        # Calculate head and tail values
        head_0 = head(0, state)
        tail_0 = tail(0, state)
        head_1 = head(1, state)
        tail_1 = tail(1, state)
        
        # Calculate R(X, T)
        if (tail_0 > t and head_1 > t) or (tail_1 > t and head_0 > t):
            r = n
        else:
            r = 0
        
        # Evaluate function
        fitness = max(tail_0, head_1) + r
    
        return fitness

class ContinuousPeaks:
    """Fitness function for Continuous Peaks optimization problem."""
    
    def __init__(self, t_pct = 0.1):
        """Initialize ContinuousPeaks object.
        
        Args:
        t_pct: float. Threshold parameter (T) for continuous peaks fitness function.
        
        Returns:
        None            
        """
        self.t_pct = t_pct
        
    def evaluate(self, state):
        """Evaluate the fitness of a state
        
        Args:
        state: array. State array for evaluation.
           
        Returns:
        fitness: float. Value of fitness function. 
        """
        n = len(state)
        t = np.ceil(self.t_pct*n)
        
        # Calculate length of maximum runs of 0's and 1's
        max_0 = max_run(0, state)
        max_1 = max_run(1, state)
        
        # Calculate R(X, T)
        if (max_0 > t and max_1 > t):
            r = n
        else:
            r = 0
        
        # Evaluate function
        fitness = max(max_0, max_1) + r
    
        return fitness
   
class Knapsack:
    """Fitness function for Knapsack optimization problem."""

    def __init__(self, weights, values, max_weight_pct = 0.35):
        """Initialize Knapsack object.
        
        Args:
        weights: array. Array of weights for each of the possible items. 
        values: array. Array of values for each of the possible items. Must be same
        length as weights array.
        max_weight_pct: float. Parameter used to set maximum capacity of knapsack (W)
        as a percentage of the total weight of all items (W = max_weight_pct*total_weight)
        
        Returns:
        None            
        """
        self.weights = weights
        self.values = values
        self.W = np.ceil(sum(self.weights)*max_weight_pct)
    
    def evaluate(self, state):
        """Evaluate the fitness of a state
        
        Args:
        state: array. State array for evaluation. Must be same length as
        weights and values arrays.
           
        Returns:
        fitness: float. Value of fitness function. 
        """
        # Calculate total weight and value of knapsack
        total_weight = np.sum(state*self.weights)
        total_value = np.sum(state*self.values)

        # Allow for weight constraint
        if total_weight <= self.W:
            fitness = total_value       
        else:
            fitness = 0
        
        return fitness

class TravellingSales:
    """Fitness function for Travelling Salesman optimization problem."""
    
    def __init__(self, distances):
        """Initialize TravellingSales object.
        
        Args:
        distances: matrix. n x n matrix giving the distances between pairs of cities.
        In most cases, we would expect the lower triangle to mirror the upper triangle 
        and the lead diagonal to be zeros.
        
        Returns:
        None            
        """
        self.distances = distances
    
    def evaluate(self, state):
        """Evaluate the fitness of a state
        
        Args:
        state: array. State array for evaluation. Must contain the same number
        of elements as the distances matrix
           
        Returns:
        fitness: float. Value of fitness function. 
        """
        # Reshape state array to match distance matrix
        state_mat = np.reshape(state, np.shape(self.distances))
        
        # Determine upper limit on distances
        max_dist = np.max(self.distances[self.distances < np.inf])*np.shape(state_mat)[0]
        
        # Replace invalid values with very large values (i.e. max_dist)
        state_mat[state_mat == -1] = max_dist 
        
        # Calculate total distance and constraint values
        total_dist = np.sum(self.distances*state_mat)

        row_sums = np.sum(state_mat, axis = 1)
        col_sums = np.sum(state_mat, axis = 0)
        diag_sum = np.sum(np.diag(state_mat))
        
        # Determine fitness
        if (np.max(row_sums) == 1 and np.min(row_sums) == 1) and \
            (np.max(col_sums) == 1 and np.min(col_sums) == 1) and \
            diag_sum == 0:
                
            fitness = max_dist - total_dist
        
        else:
            fitness = 0

        return fitness

class CustomFitness:
    """Class for generating your own fitness function."""
    
    def __init__(self, fitness_fn, **kwargs):
        """Initialize CustomFitness object.
        
        Args:
        fitness_fn: function. Function for calculating fitness of a state
        kwargs: dictionary. Additional arguments to be passed to fitness_fn
        
        Returns:
        None       
        """
        self.fitness_fn = fitness_fn
        self.kwargs = kwargs
    
    def evaluate(self, state):
        """Evaluate the fitness of a state
        
        Args:
        state: array. State array for evaluation. Must contain the same number
        of elements as the distances matrix
           
        Returns:
        fitness: float. Value of fitness function. 
        """
        fitness = self.fitness_fn(state, **self.kwargs)
        return fitness
        