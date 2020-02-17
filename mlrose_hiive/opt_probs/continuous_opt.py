""" Classes for defining optimization problem objects."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.opt_probs._opt_prob import _OptProb


class ContinuousOpt(_OptProb):
    """Class for defining continuous-state optimisation problems.

    Parameters
    ----------
    length: int
        Number of elements in state vector.

    fitness_fn: fitness function object
        Object to implement fitness function for optimization.

    maximize: bool, default: True
        Whether to maximize the fitness function.
        Set :code:`False` for minimization problem.

    min_val: float, default: 0
        Minimum value that each element of the state vector can take.

    max_val: float, default: 1
        Maximum value that each element of the state vector can take.

    step: float, default: 0.1
        Step size used in determining neighbors of current state.
    """

    def __init__(self, length, fitness_fn, maximize=True, min_val=0,
                 max_val=1, step=0.1):

        _OptProb.__init__(self, length, fitness_fn, maximize=maximize)

        if (self.fitness_fn.get_prob_type() != 'continuous') \
           and (self.fitness_fn.get_prob_type() != 'either'):
            raise Exception("fitness_fn must have problem type 'continuous'"
                            + """ or 'either'. Define problem as"""
                            + """ DiscreteOpt problem or use alternative"""
                            + """ fitness function."""
                            )

        if max_val <= min_val:
            raise Exception("""max_val must be greater than min_val.""")

        if step <= 0:
            raise Exception("""step size must be positive.""")

        if (max_val - min_val) < step:
            raise Exception("""step size must be less than"""
                            + """ (max_val - min_val).""")

        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.prob_type = 'continuous'

    def calculate_updates(self):
        """Calculate gradient descent updates.

        Returns
        -------
        updates: list
            List of back propagation weight updates.
        """
        updates = self.fitness_fn.calculate_updates()

        return updates

    def find_neighbors(self):
        """Find all neighbors of the current state."""

        self.neighbors = []

        for i in range(self.length):
            for j in [-1, 1]:
                neighbor = np.copy(self.state)
                neighbor[i] += j*self.step

                if neighbor[i] > self.max_val:
                    neighbor[i] = self.max_val

                elif neighbor[i] < self.min_val:
                    neighbor[i] = self.min_val

                if not np.array_equal(np.array(neighbor), self.state):
                    self.neighbors.append(neighbor)

    def get_prob_type(self):
        """ Return the problem type.

        Returns
        -------
        self.prob_type: string
            Returns problem type.
        """
        return self.prob_type

    def random(self):
        """Return a random state vector.

        Returns
        -------
        state: array
            Randomly generated state vector.
        """
        state = np.random.uniform(self.min_val, self.max_val, self.length)

        return state

    def random_neighbor(self):
        """Return random neighbor of current state vector.

        Returns
        -------
        neighbor: array
            State vector of random neighbor.
        """
        while True:
            neighbor = np.copy(self.state)
            i = np.random.randint(0, self.length)

            neighbor[i] += self.step*np.random.choice([-1, 1])

            if neighbor[i] > self.max_val:
                neighbor[i] = self.max_val

            elif neighbor[i] < self.min_val:
                neighbor[i] = self.min_val

            if not np.array_equal(np.array(neighbor), self.state):
                break

        return neighbor

    def random_pop(self, pop_size):
        """Create a population of random state vectors.

        Parameters
        ----------
        pop_size: int
            Size of population to be created.
        """
        if pop_size <= 0:
            raise Exception("""pop_size must be a positive integer.""")
        elif not isinstance(pop_size, int):
            if pop_size.is_integer():
                pop_size = int(pop_size)
            else:
                raise Exception("""pop_size must be a positive integer.""")

        population = []
        pop_fitness = []

        for _ in range(pop_size):
            state = self.random()
            fitness = self.eval_fitness(state)

            population.append(state)
            pop_fitness.append(fitness)

        self.population = np.array(population)
        self.pop_fitness = np.array(pop_fitness)

    def reproduce(self, parent_1, parent_2, mutation_prob=0.1):
        """Create child state vector from two parent state vectors.

        Parameters
        ----------
        parent_1: array
            State vector for parent 1.

        parent_2: array
            State vector for parent 2.

        mutation_prob: float
            Probability of a mutation at each state vector element during
            reproduction.

        Returns
        -------
        child: array
            Child state vector produced from parents 1 and 2.
        """
        if len(parent_1) != self.length or len(parent_2) != self.length:
            raise Exception("""Lengths of parents must match problem length""")

        if (mutation_prob < 0) or (mutation_prob > 1):
            raise Exception("""mutation_prob must be between 0 and 1.""")

        # Reproduce parents
        if self.length > 1:
            _n = np.random.randint(self.length - 1)
            child = np.array([0.0]*self.length)
            child[0:_n+1] = parent_1[0:_n+1]
            child[_n+1:] = parent_2[_n+1:]
        elif np.random.randint(2) == 0:
            child = np.copy(parent_1)
        else:
            child = np.copy(parent_2)

        # Mutate child
        rand = np.random.uniform(size=self.length)
        mutate = np.where(rand < mutation_prob)[0]

        for i in mutate:
            child[i] = np.random.uniform(self.min_val, self.max_val)

        return child

    def reset(self):
        """Set the current state vector to a random value and get its fitness.
        """
        self.state = self.random()
        self.fitness = self.eval_fitness(self.state)

    def update_state(self, updates):
        """Update current state given a vector of updates.

        Parameters
        ----------
        updates: array
            Update array.

        Returns
        -------
        updated_state: array
            Current state adjusted for updates.
        """
        if len(updates) != self.length:
            raise Exception("""Length of updates must match problem length""")

        updated_state = self.state + updates

        updated_state[updated_state > self.max_val] = self.max_val
        updated_state[updated_state < self.min_val] = self.min_val

        return updated_state