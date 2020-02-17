""" Functions to implement the randomized optimization and search algorithms.
"""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.decorators import short_name


def _get_hamming_distance_default(population, p1):
    hamming_distances = np.array([np.count_nonzero(p1 != p2) / len(p1) for p2 in population])
    return hamming_distances


def _get_hamming_distance_float(population, p1):
    # use squares instead?
    hamming_distances = np.array([np.abs(np.diff(p1, p2)) / len(p1) for p2 in population])
    return hamming_distances


def _genetic_alg_select_parents(pop_size, problem,
                                get_hamming_distance_func,
                                hamming_factor=0.0):
    mating_probabilities = problem.get_mate_probs()
    if get_hamming_distance_func is not None and hamming_factor > 0.01:
        selected = np.random.choice(pop_size, p=mating_probabilities)
        population = problem.get_population()
        p1 = population[selected]
        hamming_distances = get_hamming_distance_func(population, p1)
        hfa = hamming_factor / (1.0 - hamming_factor)
        hamming_distances = (hamming_distances * hfa) * mating_probabilities
        hamming_distances /= hamming_distances.sum()
        selected = np.random.choice(pop_size, p=hamming_distances)
        p2 = population[selected]

        return p1, p2

    selected = np.random.choice(pop_size,
                                size=2,
                                p=mating_probabilities)
    p1 = problem.get_population()[selected[0]]
    p2 = problem.get_population()[selected[1]]
    return p1, p2


@short_name('ga')
def genetic_alg(problem, pop_size=200, pop_breed_percent=0.75, elite_dreg_ratio=0.99,
                minimum_elites=0, minimum_dregs=0, mutation_prob=0.1,
                max_attempts=10, max_iters=np.inf, curve=False, random_state=None,
                state_fitness_callback=None, callback_user_info=None,
                hamming_factor=0.0, hamming_decay_factor=None):
    """Use a standard genetic algorithm to find the optimum for a given
    optimization problem.
    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
    pop_size: int, default: 200
        Size of population to be used in genetic algorithm.
    pop_breed_percent: float, default 0.75
        Percentage of population to breed in each iteration.
        The remainder of the population will be filled from the elite and
        dregs of the prior generation in a ratio specified by elite_dreg_ratio.
    elite_dreg_ratio: float, default:0.95
        The ratio of elites:dregs added directly to the next generation.
        For the default value, 95% of the added population will be elites,
        5% will be dregs.
    minimum_elites: int, default: 0
        Minimum number of elites to be added to next generation
    minimum_dregs: int, default: 0
        Minimum number of dregs to be added to next generation
    mutation_prob: float, default: 0.1
        Probability of a mutation at each element of the state vector
        during reproduction, expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.
    state_fitness_callback: function taking five parameters, default: None
        If specified, this callback will be invoked once per iteration.
        Parameters are (iteration, max attempts reached?, current best state, current best fit, user callback data).
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
        Numpy array of arrays containing the fitness of the entire population
        at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.
    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    breeding_pop_size = int(pop_size * pop_breed_percent) - (minimum_elites + minimum_dregs)
    if breeding_pop_size < 1:
        raise Exception("""pop_breed_percent must be large enough to ensure at least one mating.""")

    if pop_breed_percent > 1:
        raise Exception("""pop_breed_percent must be less than 1.""")

    if (elite_dreg_ratio < 0) or (elite_dreg_ratio > 1):
        raise Exception("""elite_dreg_ratio must be between 0 and 1.""")

    if (mutation_prob < 0) or (mutation_prob > 1):
        raise Exception("""mutation_prob must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    fitness_curve = []

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    if state_fitness_callback is not None:
        # initial call with base data
        state_fitness_callback(iteration=0,
                               state=problem.get_state(),
                               fitness=problem.get_adjusted_fitness(),
                               user_data=callback_user_info)
    # check for hamming
    # get_hamming_distance_default_

    get_hamming_distance_func = None
    if hamming_factor > 0:
        g1 = problem.get_population()[0][0]
        if isinstance(g1, float) or g1.dtype == 'float64':
            get_hamming_distance_func = _get_hamming_distance_float
        else:
            get_hamming_distance_func = _get_hamming_distance_default

    attempts = 0
    iters = 0

    # initialize survivor count, elite count and dreg count
    survivors_size = pop_size - breeding_pop_size
    dregs_size = max(int(survivors_size * (1.0 - elite_dreg_ratio)) if survivors_size > 1 else 0, minimum_dregs)
    elites_size = max(survivors_size - dregs_size, minimum_elites)
    if dregs_size + elites_size > survivors_size:
        over_population = dregs_size + elites_size - survivors_size
        breeding_pop_size -= over_population

    continue_iterating = True
    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Calculate breeding probabilities
        problem.eval_mate_probs()

        # Create next generation of population
        next_gen = []
        for _ in range(breeding_pop_size):
            # Select parents
            parent_1, parent_2 = _genetic_alg_select_parents(pop_size=pop_size,
                                                             problem=problem,
                                                             hamming_factor=hamming_factor,
                                                             get_hamming_distance_func=get_hamming_distance_func)

            # Create offspring
            child = problem.reproduce(parent_1, parent_2, mutation_prob)
            next_gen.append(child)

        # fill remaining population with elites/dregs
        if survivors_size > 0:
            last_gen = list(zip(problem.get_population(), problem.get_pop_fitness()))
            sorted_parents = sorted(last_gen, key=lambda f: -f[1])
            best_parents = sorted_parents[:elites_size]
            next_gen.extend([p[0] for p in best_parents])
            if dregs_size > 0:
                worst_parents = sorted_parents[-dregs_size:]
                next_gen.extend([p[0] for p in worst_parents])

        next_gen = np.array(next_gen[:pop_size])
        problem.set_population(next_gen)

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        current_fitness = problem.get_fitness()
        if next_fitness > current_fitness:
            problem.set_state(next_state)
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

        # decay hamming factor
        if hamming_decay_factor is not None and hamming_factor > 0.0:
            hamming_factor *= hamming_decay_factor
            hamming_factor = max(min(hamming_factor, 1.0), 0.0)
        # print(hamming_factor)

        # break out if requested
        if not continue_iterating:
            break
    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness, None
