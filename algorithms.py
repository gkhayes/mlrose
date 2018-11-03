""" Functions to implement the randomized optimization and search algorithms.

    Author: Genevieve Hayes
    License: 3-clause BSD license.
"""
import numpy as np

def hill_climb(problem, max_iters=np.inf, restarts=0, init_state=None):
    """Use standard hill climbing to find the optimum for a given
    optimization problem, starting from a random state.

    Args:
    problem: Optimization class object. Object containing
    optimization problem to be solved.
    max_iters: int. Maximum number of iterations of the algorithm.
    restarts: int. Number of random restarts.
    init_state: array. Numpy array containing starting state for algorithm. 
    If None then a random state is used.

    Returns:
    best_state: array. NumPy array containing state that
    optimizes fitness function.
    best_fitness: float. Value of fitness function at best state
    """

    best_fitness = -1*np.inf
    best_state = None

    for _ in range(restarts + 1):
        # Initialize optimization problem
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)
            
        iters = 0

        while iters < max_iters:
            iters += 1
            
            # Find neighbors and determine best neighbor
            problem.find_neighbors()
            next_state = problem.best_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # If best neighbor is an improvement, move to that state
            if next_fitness > problem.get_fitness():
                problem.set_state(next_state)

            else:
                break

        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()

    best_fitness = problem.get_maximize()*best_fitness
    return best_state, best_fitness


def random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=0, 
                      init_state=None):
    """Use randomized hill climbing to find the optimum for a given
    optimization problem, starting from a random state.

    Args:
    problem: Optimization class object. Object containing
    optimization problem to be solved.
    max_attempts: int. Maximum number of attempts to find a
    better neighbor at each step.
    max_iters: int. Maximum number of iterations of the algorithm.
    restarts: int. Number of random restarts.
    init_state: array. Numpy array containing starting state for algorithm. 
    If None then a random state is used.

    Returns:
    best_state: array. NumPy array containing state that optimizes
    fitness function.
    best_fitness: float. Value of fitness function at best state
    """
    best_fitness = -1*np.inf
    best_state = None

    for _ in range(restarts + 1):
        # Initialize optimization problem and attempts counter
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)
            
        attempts = 0
        iters = 0

        while (attempts < max_attempts) and (iters < max_iters):
            iters += 1
            
            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # If best neighbor is an improvement,
            # move to that state and reset attempts counter
            if next_fitness > problem.get_fitness():
                problem.set_state(next_state)
                attempts = 0

            else:
                attempts += 1

        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()

    best_fitness = problem.get_maximize()*best_fitness
    return best_state, best_fitness


def simulated_annealing(problem, schedule, max_attempts=10, max_iters=np.inf, 
                        init_state=None):
    """Use simulated annealing to find the optimum for a given
    optimization problem, starting from a random state.

    Args:
    problem: Optimization class object.
    Object containing optimization problem to be solved.
    schedule: Schedule class object. Schedule used to determine
    the value of the temperature parameter.
    max_attempts: int. Maximum number of attempts to find a better
    neighbor at each step.
    max_iters: int. Maximum number of iterations of the algorithm.
    init_state: array. Numpy array containing starting state for algorithm. 
    If None then a random state is used.

    Returns:
    best_state: array. NumPy array containing state that optimizes
    fitness function.

    best_fitness: float. Value of fitness function at best state
    """
    # Initialize problem, time and attempts counter
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)
        
    attempts = 0
    iters = 0

    while (attempts < max_attempts) and (iters < max_iters):
        temp = schedule.evaluate(iters)
        iters += 1

        if temp == 0:
            break

        else:
            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # Calculate delta E and change prob
            delta_e = next_fitness - problem.get_fitness()
            prob = np.exp(delta_e/temp)

            # If best neighbor is an improvement or random value is less 
            # than prob, move to that state and reset attempts counter
            if (delta_e > 0) or (np.random.uniform() < prob):
                problem.set_state(next_state)
                attempts = 0

            else:
                attempts += 1

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    return best_state, best_fitness


def genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10,
                max_iters=np.inf):
    """Use a standard genetic algorithm to find the optimum for a given
    optimization problem.

    Args:
    problem: Optimization class object. Object containing optimization
    problem to be solved.
    pop_size: int. Size of population to be used in genetic algorithm.
    mutation_prob: float. Probability of a mutation at each element
    during reproduction.
    max_attempts: int. Maximum number of attempts to find a better state
    at each step.
    max_iters: int. Maximum number of iterations of the algorithm.

    Returns:
    best_state: array. NumPy array containing state that optimizes
    fitness function.
    best_fitness: float. Value of fitness function at best state
    """
    # Initialize problem, population and attempts counter
    problem.reset()   
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1
        
        # Calculate breeding probabilities
        problem.eval_mate_probs()

        # Create next generation of population
        next_gen = []

        for _ in range(pop_size):
            # Select parents
            selected = np.random.choice(pop_size, size=2,
                                        p=problem.get_mate_probs())
            parent_1 = problem.get_population()[selected[0]]
            parent_2 = problem.get_population()[selected[1]]

            # Create offspring
            child = problem.reproduce(parent_1, parent_2, mutation_prob)
            next_gen.append(child)

        next_gen = np.array(next_gen)
        problem.set_population(next_gen)

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0

        else:
            attempts += 1

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    return best_state, best_fitness


def mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10, 
          max_iters=np.inf):
    """Use MIMIC to find the optimum for a given optimization problem.

    Args:
    problem: Optimization class object. Object containing optimization
    problem to be solved.
    pop_size: int. Size of population to be used in algorithm.
    keep_pct: float. Proportion of samples to keep in each iteration
    of the algorithm
    max_attempts: int. Maximum number of attempts to find a better neighbor
    at each step.
    max_iters: int. Maximum number of iterations of the algorithm.

    Returns:
    best_state: array. NumPy array containing state that optimizes
    fitness function.
    best_fitness: float. Value of fitness function at best state
    """
    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1
        
        # Get top n percent of population
        problem.find_top_pct(keep_pct)

        # Update probability estimates
        problem.eval_node_probs()

        # Generate new sample
        new_sample = problem.sample_pop(pop_size)
        problem.set_population(new_sample)

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0

        else:
            attempts += 1

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state().astype(int)

    return best_state, best_fitness
