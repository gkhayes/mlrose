""" Classes for defining neural network weight optimization problems."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.decorators import short_name
from mlrose_hiive.neural.utils import flatten_weights


@short_name('gd')
def gradient_descent(problem, max_attempts=10, max_iters=np.inf,
                     init_state=None, curve=False, random_state=None,
                     state_fitness_callback=None, callback_user_info=None):
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
    state_fitness_callback: function taking five parameters, default: None
        If specified, this callback will be invoked once per iteration.
        Parameters are (iteration, max attempts reached?, current best state, current best fit, user callback data).
        Return true to continue iterating, or false to stop.
    callback_user_info: any, default: None
        User data passed as last parameter of callback.

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

    if state_fitness_callback is not None:
        # initial call with base data
        state_fitness_callback(iteration=0,
                               state=problem.get_state(),
                               fitness=problem.get_adjusted_fitness(),
                               user_data=callback_user_info)
    fitness_curve = []
    attempts = 0
    iters = 0

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    continue_iterating = True
    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Update weights
        updates = flatten_weights(problem.calculate_updates())

        next_state = problem.update_state(updates)
        next_fitness = problem.eval_fitness(next_state)

        current_fitness = problem.get_fitness()
        if next_fitness > current_fitness:
            attempts = 0
        else:
            attempts += 1

        if curve:
            fitness_curve.append(problem.get_adjusted_fitness())

        # invoke callback
        if state_fitness_callback is not None:
            max_attempts_reached = (attempts == max_attempts) or (iters == max_iters) or problem.can_stop()
            continue_iterating = state_fitness_callback(iteration=iters,
                                                        attempt=attempts + 1,
                                                        done=max_attempts_reached,
                                                        state=problem.get_state(),
                                                        fitness=problem.get_adjusted_fitness(),
                                                        curve=np.asarray(fitness_curve) if curve else None,
                                                        user_data=callback_user_info)

        # break out if requested
        if not continue_iterating:
            break

        if next_fitness > problem.get_maximize()*best_fitness:
            best_fitness = problem.get_maximize()*next_fitness
            best_state = next_state

        problem.set_state(next_state)

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness, None
