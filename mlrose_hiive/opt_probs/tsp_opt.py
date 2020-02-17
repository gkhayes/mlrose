""" Classes for defining optimization problem objects."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.algorithms.crossovers import TSPCrossOver
from mlrose_hiive.algorithms.mutators import SwapMutator
from mlrose_hiive.fitness import TravellingSales
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt


class TSPOpt(DiscreteOpt):
    """Class for defining travelling salesperson optimisation problems.

    Parameters
    ----------
    length: int
        Number of elements in state vector. Must equal number of nodes in the
        tour.

    fitness_fn: fitness function object, default: None
        Object to implement fitness function for optimization. If :code:`None`,
        then :code:`TravellingSales(coords=coords, distances=distances)` is
        used by default.

    maximize: bool, default: False
        Whether to maximize the fitness function.
        Set :code:`False` for minimization problem.

    coords: list of pairs, default: None
        Ordered list of the (x, y) co-ordinates of all nodes. This assumes
        that travel between all pairs of nodes is possible. If this is not the
        case, then use distances instead. This argument is ignored if
        fitness_fn is not :code:`None`.

    distances: list of triples, default: None
        List giving the distances, d, between all pairs of nodes, u and v, for
        which travel is possible, with each list item in the form (u, v, d).
        Order of the nodes does not matter, so (u, v, d) and (v, u, d) are
        considered to be the same. If a pair is missing from the list, it is
        assumed that travel between the two nodes is not possible. This
        argument is ignored if fitness_fn or coords is not :code:`None`.
    """

    def __init__(self, length=None, fitness_fn=None, maximize=False, coords=None,
                 distances=None):
        if (fitness_fn is None) and (coords is None) and (distances is None):
            raise Exception("""At least one of fitness_fn, coords and"""
                            + """ distances must be specified.""")
        elif fitness_fn is None:
            fitness_fn = TravellingSales(coords=coords, distances=distances)

        if length is None:
            if coords is not None:
                length = len(coords)
            elif distances is not None:
                length = len(set([x for (x,_,_) in distances] + [x for (_,x,_) in distances]))
        self.length = length
        DiscreteOpt.__init__(self, length, fitness_fn, maximize, max_val=length,
                             crossover=TSPCrossOver(self), mutator=SwapMutator(self))

        if self.fitness_fn.get_prob_type() != 'tsp':
            raise Exception("""fitness_fn must have problem type 'tsp'.""")

        self.prob_type = 'tsp'

    def adjust_probs(self, probs):
        """Normalize a vector of probabilities so that the vector sums to 1.

        Parameters
        ----------
        probs: array
            Vector of probabilities that may or may not sum to 1.

        Returns
        -------
        adj_probs: array
            Vector of probabilities that sums to 1. Returns a zero vector if
            sum(probs) = 0.
        """
        sp = np.sum(probs)
        return np.zeros(np.shape(probs)) if sp == 0 else probs/sp

    def find_neighbors(self):
        """Find all neighbors of the current state.
        """
        self.neighbors = []

        for node1 in range(self.length - 1):
            for node2 in range(node1 + 1, self.length):
                neighbor = np.copy(self.state)

                neighbor[node1] = self.state[node2]
                neighbor[node2] = self.state[node1]
                self.neighbors.append(neighbor)

    def random(self):
        """Return a random state vector.

        Returns
        -------
        state: array
            Randomly generated state vector.
        """
        state = np.random.permutation(self.length)

        return state

    def random_mimic(self):
        """Generate single MIMIC sample from probability density.

        Returns
        -------
        state: array
            State vector of MIMIC random sample.
        """
        remaining = list(np.arange(self.length))
        state = np.zeros(self.length, dtype=np.int8)
        sample_order = self.sample_order[1:]
        node_probs = np.copy(self.node_probs)

        # Get value of first element in new sample
        state[0] = np.random.choice(self.length, p=node_probs[0, 0])
        remaining.remove(state[0])
        node_probs[:, :, state[0]] = 0

        # Get sample order
        self.find_sample_order()
        sample_order = self.sample_order[1:]

        # Set values of remaining elements of state
        for i in sample_order:
            par_ind = self.parent_nodes[i-1]
            par_value = state[par_ind]
            probs = node_probs[i, par_value]

            if np.sum(probs) == 0:
                next_node = np.random.choice(remaining)

            else:
                adj_probs = self.adjust_probs(probs)
                next_node = np.random.choice(self.length, p=adj_probs)

            state[i] = next_node
            remaining.remove(next_node)
            node_probs[:, :, next_node] = 0

        return state

    def random_neighbor(self):
        """Return random neighbor of current state vector.

        Returns
        -------
        neighbor: array
            State vector of random neighbor.
        """
        neighbor = np.copy(self.state)
        node1, node2 = np.random.choice(np.arange(self.length),
                                        size=2, replace=False)

        neighbor[node1] = self.state[node2]
        neighbor[node2] = self.state[node1]

        return neighbor

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

        self.find_sample_order()
        new_sample = []

        for _ in range(sample_size):
            state = self.random_mimic()
            new_sample.append(state)

        new_sample = np.array(new_sample)

        return new_sample
