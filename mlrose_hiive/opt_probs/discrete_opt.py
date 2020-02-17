""" Classes for defining optimization problem objects."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_tree

from mlrose_hiive.algorithms.crossovers import UniformCrossOver
from mlrose_hiive.algorithms.mutators import SwapMutator
from mlrose_hiive.opt_probs._opt_prob import _OptProb


class DiscreteOpt(_OptProb):
    """Class for defining discrete-state optimization problems.

    Parameters
    ----------
    length: int
        Number of elements in state vector.

    fitness_fn: fitness function object
        Object to implement fitness function for optimization.

    maximize: bool, default: True
        Whether to maximize the fitness function.
        Set :code:`False` for minimization problem.

    max_val: int, default: 2
        Number of unique values that each element in the state vector
        can take. Assumes values are integers in the range 0 to
        (max_val - 1), inclusive.
    """

    def __init__(self, length, fitness_fn, maximize=True, max_val=2,
                 crossover=None, mutator=None):
        self._get_mutual_info_impl = self._get_mutual_info_slow

        _OptProb.__init__(self, length, fitness_fn, maximize)

        if self.fitness_fn.get_prob_type() == 'continuous':
            raise Exception("""fitness_fn must have problem type 'discrete',"""
                            + """ 'either' or 'tsp'. Define problem as"""
                            + """ ContinuousOpt problem or use alternative"""
                            + """ fitness function."""
                            )

        if max_val < 0:
            raise Exception("""max_val must be a positive integer.""")
        elif not isinstance(max_val, int):
            if max_val.is_integer():
                self.max_val = int(max_val)
            else:
                raise Exception("""max_val must be a positive integer.""")
        else:
            self.max_val = max_val

        self.keep_sample = []
        self.node_probs = np.zeros([self.length, self.max_val, self.max_val])
        self.parent_nodes = []
        self.sample_order = []
        self.prob_type = 'discrete'
        self.noise = 0

        self._crossover = UniformCrossOver(self) if crossover is None else crossover
        self._mutator = SwapMutator(self) if mutator is None else mutator

        self._mut_mask = None
        self._mut_inf = None

    def eval_node_probs(self):
        """Update probability density estimates.
        """
        # Create mutual info matrix
        mutual_info = self._get_mutual_info_impl()

        # Find minimum spanning tree of mutual info matrix
        csr_mx = csr_matrix(mutual_info)
        mst = minimum_spanning_tree(csr_mx)

        # Convert minimum spanning tree to depth first tree with node 0 as root
        dft = depth_first_tree(csr_matrix(mst.toarray()), 0, directed=False)
        dft = np.round(dft.toarray(), 10)

        # Determine parent of each node
        parent = np.argmin(dft[:, 1:], axis=0)

        # Get probs
        probs = np.zeros([self.length, self.max_val, self.max_val])

        probs[0, :] = np.histogram(self.keep_sample[:, 0],
                                   np.arange(self.max_val + 1),
                                   density=True)[0]

        for i in range(1, self.length):
            for j in range(self.max_val):
                subset = self.keep_sample[np.where(
                    self.keep_sample[:, parent[i - 1]] == j)[0]]

                if not len(subset):
                    probs[i, j] = 1 / self.max_val
                else:
                    temp_probs = np.histogram(subset[:, i],
                                              np.arange(self.max_val + 1),
                                              density=True)[0]

                    # Check if noise argument is not default (in epsilon)
                    if self.noise > 0:
                        # Add noise, from the mimic argument "noise"
                        temp_probs = (temp_probs + self.noise)
                        # All probability adds up to one
                        temp_probs = np.divide(temp_probs, np.sum(temp_probs))
                        # Handle floating point error to ensure probability adds up to 1
                        if sum(temp_probs) != 1.0:
                            temp_probs = np.divide(temp_probs, np.sum(temp_probs))
                    # Set probability
                    probs[i, j] = temp_probs

        # Update probs and parent
        self.node_probs = probs
        self.parent_nodes = parent

    def set_mimic_fast_mode(self, fast_mode):
        if fast_mode:
            mut_mask = np.zeros([self.length, self.length], dtype=bool)
            for i in range(0, self.length):
                for j in range(i, self.length):
                    mut_mask[i, j] = True
            mut_mask = mut_mask.reshape((self.length * self.length))
            self._mut_mask = mut_mask
            # Set ignore error to ignore dividing by zero
            np.seterr(divide='ignore', invalid='ignore')
            self._get_mutual_info_impl = self._get_mutual_info_fast
            self._mut_inf = np.zeros([self.length * self.length])
        else:
            self._mut_mask = None
            self._get_mutual_info_impl = self._get_mutual_info_slow
            self._mut_inf = None

    def _get_mutual_info_slow(self):
        mutual_info = np.zeros([self.length, self.length])
        for i in range(self.length - 1):
            for j in range(i + 1, self.length):
                mutual_info[i, j] = -1 * mutual_info_score(
                    self.keep_sample[:, i],
                    self.keep_sample[:, j])
        return mutual_info

    # adapted from https://github.com/parkds/mlrose/blob/f7154a1d3e3fdcd934bb3c683b943264d2870fd1/mlrose/algorithms.py
    # (thanks to David Sejin Park)
    def _get_mutual_info_fast(self):
        if self._mut_inf is None:
            # restore sanity
            self._get_mutual_info_impl = self._get_mutual_info_slow
            return self._get_mutual_info_impl()

        # get length of the sample which survived from mimic iteration
        len_sample_kept = self.keep_sample.shape[0]
        # get the length of the bit sequence / problem size
        len_prob = self.keep_sample.shape[1]

        # Expand the matrices to so each row corresponds to a row by row combination of the list of samples
        b = np.repeat(self.keep_sample, self.length).reshape(len_sample_kept, len_prob * len_prob)
        d = np.hstack(([self.keep_sample] * len_prob))

        # Compute the mutual information matrix in bulk, by iterating through the list of
        # possible feature values ((max_val-1)^2).
        # For example, a binary string would go through 00 01 10 11, for a total of 4 iterations.

        # First initialize the mutual info matrix.
        self._mut_inf.fill(0)
        # Pre-compute the U and V which gets computed multiple times in the inner loop.
        U = {}
        V = {}
        U_sum = {}
        V_sum = {}
        for i in range(0, self.max_val):
            U[i] = (d == i)
            V[i] = (b == i)
            U_sum[i] = np.sum(d == i, axis=0)
            V_sum[i] = np.sum(b == i, axis=0)

        # Compute the mutual information for all sample to sample combination for each feature combination
        # ((max_val-1)^2)
        for i in range(0, self.max_val):
            for j in range(0, self.max_val):
                # This corresponds to U and V of mutual info matrix, for this feature pair
                coeff = np.sum(U[i] * V[j], axis=0)
                # Compute length N, for the particular feature pair
                UV_length = (U_sum[i] * V_sum[j])

                # compute the second term of the MI matrix
                temp = np.log(coeff) - np.log(UV_length) + np.log(len_sample_kept)
                # remove the nans and negative infinity
                temp[np.isnan(temp)] = 0
                temp[np.isneginf(temp)] = 0

                # combine the first and the second term, divide by the length N.
                # Add the whole MI matrix for the feature to the previously computed values
                div = temp * np.divide(coeff, len_sample_kept)
                div[self._mut_mask] = 0
                self._mut_inf += div

        # Need to multiply by negative to get the mutual information
        self._mut_inf = -self._mut_inf.reshape(self.length, self.length)
        # Only get the upper triangle matrix above the identity row.
        # Possible enhancements, currently we are doing double the computation required.
        # Pre set the matrix so the computation is only done for rows that are needed. To do for the future.

        mutual_info = self._mut_inf.T
        self._mut_inf = self._mut_inf.reshape(self.length * self.length)
        return mutual_info

    def find_neighbors(self):
        """Find all neighbors of the current state.
        """
        self.neighbors = []

        if self.max_val == 2:
            for i in range(self.length):
                neighbor = np.copy(self.state)
                neighbor[i] = np.abs(neighbor[i] - 1)
                self.neighbors.append(neighbor)

        else:
            for i in range(self.length):
                vals = list(np.arange(self.max_val))
                vals.remove(self.state[i])

                for j in vals:
                    neighbor = np.copy(self.state)
                    neighbor[i] = j
                    self.neighbors.append(neighbor)

    def find_sample_order(self):
        """Determine order in which to generate sample vector elements.
        """
        sample_order = []
        last = [0]
        parent = np.array(self.parent_nodes)

        while len(sample_order) < self.length:
            inds = []

            # If last nodes list is empty, select random node than has not
            # previously been selected
            if len(last) == 0:
                inds = [np.random.choice(list(set(np.arange(self.length)) -
                                              set(sample_order)))]
            else:
                for i in last:
                    inds += list(np.where(parent == i)[0] + 1)

            sample_order += last
            last = inds

        self.sample_order = sample_order

    def find_top_pct(self, keep_pct):
        """Select samples with fitness in the top keep_pct percentile.

        Parameters
        ----------
        keep_pct: float
            Proportion of samples to keep.
        """
        if (keep_pct < 0) or (keep_pct > 1):
            raise Exception("""keep_pct must be between 0 and 1.""")

        # Determine threshold
        theta = np.percentile(self.pop_fitness, 100 * (1 - keep_pct))

        # Determine samples for keeping
        keep_inds = np.where(self.pop_fitness >= theta)[0]

        # Determine sample for keeping
        self.keep_sample = self.population[keep_inds]

    def get_keep_sample(self):
        """ Return the keep sample.

        Returns
        -------
        self.keep_sample: array
            Numpy array containing samples with fitness in the top keep_pct
            percentile.
        """
        return self.keep_sample

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
        state = np.random.randint(0, self.max_val, self.length)

        return state

    def random_neighbor(self):
        """Return random neighbor of current state vector.

        Returns
        -------
        neighbor: array
            State vector of random neighbor.
        """
        neighbor = np.copy(self.state)
        i = np.random.randint(0, self.length)

        if self.max_val == 2:
            neighbor[i] = np.abs(neighbor[i] - 1)

        else:
            vals = list(np.arange(self.max_val))
            vals.remove(neighbor[i])
            neighbor[i] = vals[np.random.randint(0, self.max_val - 1)]

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
            population.append(state)
            fitness = self.eval_fitness(state)
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
            Probability of a mutation at each state element during
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
        child = self._crossover.mate(parent_1, parent_2)

        # Mutate child
        child = self._mutator.mutate(child, mutation_prob)

        return child

    def reset(self):
        """Set the current state vector to a random value and get its fitness.
        """
        self.state = self.random()
        self.fitness = self.eval_fitness(self.state)

    def sample_pop(self, sample_size):
        """Generate new sample from probability density.

        Parameters
        ----------
        sample_size: int
            Size of sample to be generated.

        Returns
        -------
        new_sample: array
            Numpy array containing new sample.
        """
        if sample_size <= 0:
            raise Exception("""sample_size must be a positive integer.""")
        elif not isinstance(sample_size, int):
            if sample_size.is_integer():
                sample_size = int(sample_size)
            else:
                raise Exception("""sample_size must be a positive integer.""")

        # Initialize new sample matrix
        new_sample = np.zeros([sample_size, self.length])

        # Get value of first element in new samples
        new_sample[:, 0] = np.random.choice(self.max_val, sample_size,
                                            p=self.node_probs[0, 0])

        # Get sample order
        self.find_sample_order()
        sample_order = self.sample_order[1:]

        # Get values for remaining elements in new samples
        for i in sample_order:
            par_ind = self.parent_nodes[i - 1]

            for j in range(self.max_val):
                inds = np.where(new_sample[:, par_ind] == j)[0]
                new_sample[inds, i] = np.random.choice(self.max_val,
                                                       len(inds),
                                                       p=self.node_probs[i, j])

        return new_sample
