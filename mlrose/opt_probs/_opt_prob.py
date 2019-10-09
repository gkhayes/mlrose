""" Classes for defining optimization problem objects."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np


class _OptProb:
    """Base class for optimisation problems.

    Parameters
    ----------
    length: int
        Number of elements in state vector.
    fitness_fn: fitness function object
        Object to implement fitness function for optimization.
    maximize: bool, default: True
        Whether to maximize the fitness function.
        Set :code:`False` for minimization problem.
    """

    def __init__(self, length, fitness_fn, maximize=True):

        if length < 0:
            raise Exception("""length must be a positive integer.""")
        elif not isinstance(length, int):
            if length.is_integer():
                self.length = int(length)
            else:
                raise Exception("""length must be a positive integer.""")
        else:
            self.length = length

        self.state = np.array([0]*self.length)
        self.neighbors = []
        self.fitness_fn = fitness_fn
        self.fitness = 0
        self.population = []
        self.pop_fitness = []
        self.mate_probs = []

        if maximize:
            self.maximize = 1.0
        else:
            self.maximize = -1.0

    def best_child(self):
        """Return the best state in the current population.

        Returns
        -------
        best: array
            State vector defining best child.
        """
        best = self.population[np.argmax(self.pop_fitness)]

        return best

    def best_neighbor(self):
        """Return the best neighbor of current state.

        Returns
        -------
        best: array
            State vector defining best neighbor.
        """
        fitness_list = []

        for neigh in self.neighbors:
            fitness = self.eval_fitness(neigh)
            fitness_list.append(fitness)

        best = self.neighbors[np.argmax(fitness_list)]

        return best

    def eval_fitness(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State vector for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """
        if len(state) != self.length:
            raise Exception("state length must match problem length")

        fitness = self.maximize*self.fitness_fn.evaluate(state)

        return fitness

    def eval_mate_probs(self):
        """
        Calculate the probability of each member of the population reproducing.
        """
        pop_fitness = np.copy(self.pop_fitness)

        # Set -1*inf values to 0 to avoid dividing by sum of infinity.
        # This forces mate_probs for these pop members to 0.
        pop_fitness[pop_fitness == -1.0*np.inf] = 0

        # account for maximize = False
        if self.maximize == -1:
            pop_fitness -= np.min(pop_fitness)

        if np.sum(pop_fitness) == 0:
            self.mate_probs = np.ones(len(pop_fitness)) \
                              / len(pop_fitness)
        else:
            self.mate_probs = pop_fitness/np.sum(pop_fitness)

    def get_fitness(self):
        """ Return the fitness of the current state vector.

        Returns
        -------
        self.fitness: float
            Fitness value of current state vector.
        """
        return self.fitness

    def get_adjusted_fitness(self):
        """ Return maximization factor * fitness of the current state vector.

        Returns
        -------
        self.maximize*self.fitness: float
            Fitness value of current state vector adjusted by maximization factor.
        """
        return self.maximize * self.fitness

    def get_length(self):
        """ Return the state vector length.

        Returns
        -------
        self.length: int
            Length of state vector.
        """
        return self.length

    def get_mate_probs(self):
        """ Return the population mate probabilities.

        Returns
        -------
        self.mate_probs: array.
            Numpy array containing mate probabilities of the current
            population.
        """
        return self.mate_probs

    def get_maximize(self):
        """ Return the maximization multiplier.

        Returns
        -------
        self.maximize: int
            Maximization multiplier.
        """
        return self.maximize

    def get_pop_fitness(self):
        """ Return the current population fitness array.

        Returns
        -------
        self.pop_fitness: array
            Numpy array containing the fitness values for the current
            population.
        """
        return self.pop_fitness

    def get_population(self):
        """ Return the current population.

        Returns
        -------
        self.population: array
            Numpy array containing current population.
        """
        return self.population

    def get_state(self):
        """ Return the current state vector.

        Returns
        -------
        self.state: array
            Current state vector.
        """
        return self.state

    def set_population(self, new_population):
        """ Change the current population to a specified new population and get
        the fitness of all members.

        Parameters
        ----------
        new_population: array
            Numpy array containing new population.
        """
        self.population = new_population
        self.evaluate_population_fitness()

    def evaluate_population_fitness(self):
        # Calculate fitness
        pop_fitness = []

        for i in range(len(self.population)):
            fitness = self.eval_fitness(self.population[i])
            pop_fitness.append(fitness)

        self.pop_fitness = np.array(pop_fitness)

    def set_state(self, new_state):
        """
        Change the current state vector to a specified value
        and get its fitness.

        Parameters
        ----------
        new_state: array
            New state vector value.
        """
        if len(new_state) != self.length:
            raise Exception("""new_state length must match problem length""")

        self.state = new_state
        self.fitness = self.eval_fitness(self.state)

    def can_stop(self):
        return False
