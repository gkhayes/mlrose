""" Functions to implement the randomized optimization and search algorithms.

    Author: Genevieve Hayes <ghayes17@gmail.com>
    License: 3-clause BSD license.
"""

import numpy as np

def hill_climb(problem, restarts = 1):
    """Use standard hill climbing to find the maximum for a given 
    optimization problem, starting from a random state.

    Args:
    problem: Optimization class object. Object containing optimization problem to be solved.
    restarts: int. Number of random restarts.
       
    Returns:
    best_state: array. NumPy array containing state that optimizes fitness function.
    best_fitness: float. Value of fitness function at best state
    """
    
    best_fitness = -1*np.inf
    best_state = None
    
    for i in range(restarts):
        # Initialize optimization problem
        problem.reset()
        
        while True:
            # Find neighbors and determine best neighbor
            problem.get_neighbors()
            next_state = problem.best_neighbor()
            next_fitness = problem.get_fitness(next_state)
            
            # If best neighbor is an improvement, move to that state
            if next_fitness > problem.fitness:
                problem.change_string(next_state)
            
            else:
                break
        
        # Update best state and best fitness
        if problem.fitness > best_fitness:
            best_fitness = problem.fitness
            best_state = problem.state      
    
    return best_state, best_fitness

def random_hill_climb(problem, max_attempts = 10, restarts = 1):
    """Use randomized hill climbing to find the maximum for a given 
    optimization problem, starting from a random state.
    
    Args:
    problem: Optimization class object. Object containing optimization problem to be solved.
    max_attempts: int. Maximum number of attempts to find a better neighbor at each step.
    restarts: int. Number of random restarts.
       
    Returns:
    best_state: array. NumPy array containing state that optimizes fitness function.
    best_fitness: float. Value of fitness function at best state
    """
    
    best_fitness = -1*np.inf
    best_state = None
    
    for i in range(restarts):
        # Initialize optimization problem and attempts counter
        problem.reset()
        attempts = 0
        
        while attempts < max_attempts:
            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.get_fitness(next_state)
            
            # If best neighbor is an improvement, move to that state and reset attempts counter
            if next_fitness > problem.fitness:
                problem.change_string(next_state)
                attempts = 0
                
            else:
                attempts += 1
        
        # Update best state and best fitness        
        if problem.fitness > best_fitness:
            best_fitness = problem.fitness
            best_state = problem.state
    
    return best_state, best_fitness