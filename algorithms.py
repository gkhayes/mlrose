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
            problem.find_neighbors()
            next_state = problem.best_neighbor()
            next_fitness = problem.calculate_fitness(next_state)
            
            # If best neighbor is an improvement, move to that state
            if next_fitness > problem.get_fitness():
                problem.set_state(next_state)
            
            else:
                break
        
        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()  
    
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
            next_fitness = problem.calculate_fitness(next_state)
            
            # If best neighbor is an improvement, move to that state and reset attempts counter
            if next_fitness > problem.get_fitness():
                problem.set_state(next_state)
                attempts = 0
                
            else:
                attempts += 1
        
        # Update best state and best fitness        
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()
    
    return best_state, best_fitness

class GeomDecay:
    """Schedule for geometrically decaying the simulated annealing temperature parameter T"""
    
    def __init__(self, init_t = 1.0, decay = 0.99, min_t = 0.001):
        """Initialize decay schedule object
        
        Args: 
        init_t: float. Initial value of temperature parameter T
        decay: float. Temperature decay parameter.
        min_t: float. Minimum value of temperature parameter.
        
        Returns:
        None
        """
        self.init_t = init_t
        self.decay = decay
        self.min_t = min_t
        
    def evaluate(self, t):
        """Evaluate the temperature parameter at time t
        
        Args:    
        t: int. Time at which the temperature paramter t is evaluated
        
        Returns:
        temp: float. Temperature parameter at time t
        """
        
        temp = self.init_t*(self.decay**t)
        
        if temp < self.min_t:
            temp = self.min_t
        
        return temp

class ArithDecay:
    """Schedule for arithmetically decaying the simulated annealing temperature parameter T"""
    
    def __init__(self, init_t = 1.0, decay = 0.0001, min_t = 0.001):
        """Initialize decay schedule object
        
        Args: 
        init_t: float. Initial value of temperature parameter T
        decay: float. Temperature decay parameter.
        min_t: float. Minimum value of temperature parameter.
        
        Returns:
        None
        """
        self.init_t = init_t
        self.decay = decay
        self.min_t = min_t
        
    def evaluate(self, t):
        """Evaluate the temperature parameter at time t
        
        Args:    
        t: int. Time at which the temperature paramter t is evaluated
        
        Returns:
        temp: float. Temperature parameter at time t
        """
        
        temp = self.init_t - (self.decay*t)
        
        if temp < self.min_t:
            temp = self.min_t
        
        return temp

class ExpDecay:
    """Schedule for exponentially decaying the simulated annealing temperature parameter T"""
    
    def __init__(self, init_t = 1.0, exp_const = 0.005, min_t = 0.001):
        """Initialize decay schedule object
        
        Args: 
        init_t: float. Initial value of temperature parameter T
        exp_const: float. Exponential constant parameter.
        min_t: float. Minimum value of temperature parameter.
        
        Returns:
        None
        """
        self.init_t = init_t
        self.exp_const = exp_const
        self.min_t = min_t
        
    def evaluate(self, t):
        """Evaluate the temperature parameter at time t
        
        Args:    
        t: int. Time at which the temperature paramter t is evaluated
        
        Returns:
        temp: float. Temperature parameter at time t
        """
        
        temp = self.init_t*np.exp(-1.0*self.exp_const*t)
        
        if temp < self.min_t:
            temp = self.min_t
        
        return temp   

def simulated_annealing(problem, schedule, max_attempts = 10):
    """Use simulated annealing to find the maximum for a given 
    optimization problem, starting from a random state.
    
    Args:
    problem: Optimization class object. Object containing optimization problem to be solved.
    schedule: Schedule class object. Schedule used to determine the value of the temperature parameter.
    max_attempts: int. Maximum number of attempts to find a better neighbor at each step.
       
    Returns:
    best_state: array. NumPy array containing state that optimizes fitness function.
    best_fitness: float. Value of fitness function at best state
    """
    # Initialize problem, time and attempts counter
    problem.reset()
    t = 0
    attempts = 0
    
    while attempts < max_attempts:
        temp = schedule.evaluate(t)
        
        if temp == 0:
            break
        
        else:
            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.calculate_fitness(next_state)
            
            # Calculate delta E and change prob
            delta_e = next_fitness - problem.get_fitness()
            prob = np.exp(delta_e/temp)
            
            # If best neighbor is an improvement or random value is less than prob,
            # move to that state and reset attempts counter
            if (delta_e > 0) or (np.random.uniform() < prob):
                problem.set_state(next_state)
                attempts = 0
                
            else:
                attempts += 1
            
            # Increment time counter
            t += 1
        
    best_fitness = problem.get_fitness()
    best_state = problem.get_state()
    
    return best_state, best_fitness

def reproduce(parent_1, parent_2, mutation_prob = 0.1):
    """Create child bitstring from two parent bitstrings.
    
    Args:
    parent_1: array. Bitstring for parent 1.
    parent_2: array. Bitstring for parent 2.
    mutation_prob: float. Probability of a mutation at each bit during reproduction.
    
    Returns:
    child: array. Child bitstring produced from parents 1 and 2.
    """
    # Reproduce parents
    n = np.random.randint(len(parent_1)-1)
    child = np.array([0]*len(parent_1))
    child[0:n+1] = parent_1[0:n+1]
    child[n+1:] = parent_2[n+1:]
    
    # Mutate child
    rand = np.random.uniform(size = len(child))
    mutate = np.where(rand < mutation_prob)[0]

    for i in mutate:
        child[i] = np.abs(child[i] - 1)
        
    return child 
    
def genetic_alg(problem, pop_size, mutation_prob, max_attempts):
    """Use a standard genetic algorithm to find the maximum for a given 
    optimization problem.
    
    Args:
    problem: Optimization class object. Object containing optimization problem to be solved.
    pop_size: int. Size of population to be used in genetic algorithm.
    mutation_prob: float. Probability of a mutation at each bit during reproduction.
    max_attempts: int. Maximum number of attempts to find a better neighbor at each step.
       
    Returns:
    best_state: array. NumPy array containing state that optimizes fitness function.
    best_fitness: float. Value of fitness function at best state
    """
    # Initialize problem, population and attempts counter
    problem.reset()
    problem.create_population(pop_size)
    attempts = 0
    
    while attempts < max_attempts:
        # Calculate breeding probabilities
        problem.calculate_probs()
        
        # Create next generation of population
        next_gen = []
        
        for i in range(pop_size):
            # Select parents
            selected = np.random.choice(pop_size, size = 2, p = problem.probs)
            parent_1 = problem.get_population()[selected[0]]
            parent_2 = problem.get_population()[selected[1]]
            
            # Create offspring
            child = reproduce(parent_1, parent_2, mutation_prob)
            next_gen.append(child)
        
        next_gen = np.array(next_gen)
        problem.set_population(next_gen)
        
        next_state = problem.best_child()
        next_fitness = problem.calculate_fitness(next_state)
        
        # If best child is an improvement, move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0
                
        else:
            attempts += 1
        
    best_fitness = problem.get_fitness()
    best_state = problem.get_state()
    
    return best_state, best_fitness

def mimic(problem, pop_size, keep_pct, max_attempts):
    """Use MIMIC to find the maximum for a given optimization problem.
    
    Args:
    problem: Optimization class object. Object containing optimization problem to be solved.
    pop_size: int. Size of population to be used in algorithm.
    keep_pct: float. Proportion of samples to keep in each iteration of the algorithm
    max_attempts: int. Maximum number of attempts to find a better neighbor at each step.
       
    Returns:
    best_state: array. NumPy array containing state that optimizes fitness function.
    best_fitness: float. Value of fitness function at best state
    """
    # Initialize problem, population and attempts counter
    problem.reset()
    problem.create_population(pop_size)
    attempts = 0
    
    while attempts < max_attempts:
        # Get top n percent of population
        problem.select_keep_sample(keep_pct)
        
        # Update probability estimates
        problem.update_probs()
        
        # Generate new sample
        new_sample = problem.generate_new_sample(pop_size)
        problem.set_population(new_sample)
        
        next_state = problem.best_child()
        next_fitness = problem.calculate_fitness(next_state)
        
        # If best child is an improvement, move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0
                
        else:
            attempts += 1
        
    best_fitness = problem.get_fitness()
    best_state = problem.get_state().astype(int)
    
    return best_state, best_fitness