""" Functions to implement the randomized optimization and search algorithms.
"""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.decorators import short_name


@short_name('hc')
def hill_climb(problem, max_iters=np.inf, restarts=0, init_state=None,
               curve=False, random_state=None,
               state_fitness_callback=None, callback_user_info=None):
    """Use standard hill climbing to find the optimum for a given
    optimization problem.
    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm for each restart.
    restarts: int, default: 0
        Number of random restarts.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.
    state_fitness_callback: function taking two parameters, default: None
        If specified, this callback will be invoked once per iteration.
        Parameters are (iteration, current best state, current best fit, user callback data).
        Return true to continue iterating, or false to stop.
    callback_user_info: any, default: None
        User data passed as last parameter of callback.
    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array containing the fitness at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.
    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if (not isinstance(restarts, int) and not restarts.is_integer()) \
       or (restarts < 0):
        raise Exception("""restarts must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    best_fitness = -np.inf
    best_state = None

    fitness_curve = []
    best_fitness_curve = []

    continue_iterating = True
    for current_restart in range(restarts + 1):
        # Initialize optimization problem
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)

        callback_extra_data = None
        if state_fitness_callback is not None:
            callback_extra_data = callback_user_info + [('current_restart', current_restart)]
            # initial call with base data
            state_fitness_callback(iteration=0,
                                   state=problem.get_state(),
                                   fitness=problem.get_adjusted_fitness(),
                                   user_data=callback_extra_data)
        iters = 0
        while iters < max_iters:
            iters += 1

            # Find neighbors and determine best neighbor
            problem.find_neighbors()
            next_state = problem.best_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            if curve:
                fitness_curve.append(problem.get_adjusted_fitness())

            # invoke callback
            if state_fitness_callback is not None:
                max_attempts_reached = (iters == max_iters) or problem.can_stop()
                continue_iterating = state_fitness_callback(iteration=iters,
                                                            attempt=None,
                                                            done=max_attempts_reached,
                                                            state=problem.get_state(),
                                                            fitness=problem.get_adjusted_fitness(),
                                                            curve=np.asarray(fitness_curve) if curve else None,
                                                            user_data=callback_extra_data)
                # break out if requested
                if not continue_iterating:
                    break

            # If best neighbor is an improvement, move to that state
            current_fitness = problem.get_fitness()
            if next_fitness > current_fitness:
                problem.set_state(next_state)
            else:
                break

        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()
            if curve:
                best_fitness_curve = [*fitness_curve]
                fitness_curve = []

        # break out if requested
        if not continue_iterating:
            break

    best_fitness = problem.get_maximize()*best_fitness

    if curve:
        return best_state, best_fitness, np.asarray(best_fitness_curve)

    return best_state, best_fitness, None
