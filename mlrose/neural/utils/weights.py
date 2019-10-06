""" Classes for defining neural network weight optimization problems."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np


def flatten_weights(weights):
    """Flatten list of weights arrays into a 1D array.

    Parameters
    ----------
    weights: list of arrays
        List of 2D arrays for flattening.

    Returns
    -------
    flat_weights: array
        1D weights array.
    """
    flat_weights = []

    for i in range(len(weights)):
        flat_weights += list(weights[i].flatten())

    flat_weights = np.array(flat_weights)

    return flat_weights


def unflatten_weights(flat_weights, node_list):
    """Convert 1D weights array into list of 2D arrays.

    Parameters
    ----------
    flat_weights: array
        1D weights array.

    node_list: list
        List giving the number of nodes in each layer of the network,
        including the input and output layers.

    Returns
    -------
    weights: list of arrays
        List of 2D arrays created from flat_weights.
    """
    nodes = 0
    for i in range(len(node_list) - 1):
        nodes += node_list[i]*node_list[i + 1]

    if len(flat_weights) != nodes:
        raise Exception("""flat_weights must have length %d""" % (nodes,))

    weights = []
    start = 0

    for i in range(len(node_list) - 1):
        end = start + node_list[i]*node_list[i + 1]
        weights.append(np.reshape(flat_weights[start:end],
                                  [node_list[i], node_list[i+1]]))
        start = end

    return weights


def gradient_descent_original(problem, max_attempts=10, max_iters=np.inf,
                     init_state=None, curve=False, random_state=None):
    """Use gradient_descent to find the optimal neural network weights.
    Parameters
    ----------
    problem: optimization object
        Object containing optimization problem to be solved.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    init_state: array, default: None
        Numpy array containing starting state for algorithm.
        If None, then a random state is used.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array containing the fitness at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.
    """
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    # Initialize problem, time and attempts counter
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    if curve:
        fitness_curve = []

    attempts = 0
    iters = 0

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Update weights
        updates = flatten_weights(problem.calculate_updates())

        next_state = problem.update_state(updates)
        next_fitness = problem.eval_fitness(next_state)

        if next_fitness > problem.get_fitness():
            attempts = 0
        else:
            attempts += 1

        if next_fitness > problem.get_maximize()*best_fitness:
            best_fitness = problem.get_maximize()*next_fitness
            best_state = next_state

        if curve:
            fitness_curve.append(problem.get_fitness())

        problem.set_state(next_state)

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness, None
