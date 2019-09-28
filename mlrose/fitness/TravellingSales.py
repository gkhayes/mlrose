""" Classes for defining fitness functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np


class TravellingSales:
    """Fitness function for Travelling Salesman optimization problem.
    Evaluates the fitness of a tour of n nodes, represented by state vector
    :math:`x`, giving the order in which the nodes are visited, as the total
    distance travelled on the tour (including the distance travelled between
    the final node in the state vector and the first node in the state vector
    during the return leg of the tour). Each node must be visited exactly
    once for a tour to be considered valid.

    Parameters
    ----------
    coords: list of pairs, default: None
        Ordered list of the (x, y) coordinates of all nodes (where element i
        gives the coordinates of node i). This assumes that travel between
        all pairs of nodes is possible. If this is not the case, then use
        :code:`distances` instead.

    distances: list of triples, default: None
        List giving the distances, d, between all pairs of nodes, u and v, for
        which travel is possible, with each list item in the form (u, v, d).
        Order of the nodes does not matter, so (u, v, d) and (v, u, d) are
        considered to be the same. If a pair is missing from the list, it is
        assumed that travel between the two nodes is not possible. This
        argument is ignored if coords is not :code:`None`.

    Examples
    --------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose
        >>> import numpy as np
        >>> coords = [(0, 0), (3, 0), (3, 2), (2, 4), (1, 3)]
        >>> dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6),
                     (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        >>> fitness_coords = mlrose.TravellingSales(coords=coords)
        >>> state = np.array([0, 1, 4, 3, 2])
        >>> fitness_coords.evaluate(state)
        13.86138...
        >>> fitness_dists = mlrose.TravellingSales(distances=dists)
        >>> fitness_dists.evaluate(state)
        29

    Note
    ----
    1. The TravellingSales fitness function is suitable for use in travelling
       salesperson (tsp) optimization problems *only*.
    2. It is necessary to specify at least one of :code:`coords` and
       :code:`distances` in initializing a TravellingSales fitness function
       object.
    """

    def __init__(self, coords=None, distances=None):

        if coords is None and distances is None:
            raise Exception("""At least one of coords and distances must be"""
                            + """ specified.""")

        elif coords is not None:
            self.is_coords = True
            path_list = []
            dist_list = []

        else:
            self.is_coords = False

            # Remove any duplicates from list
            distances = list({tuple(sorted(dist[0:2]) + [dist[2]])
                              for dist in distances})

            # Split into separate lists
            node1_list, node2_list, dist_list = zip(*distances)

            if min(dist_list) <= 0:
                raise Exception("""The distance between each pair of nodes"""
                                + """ must be greater than 0.""")
            if min(node1_list + node2_list) < 0:
                raise Exception("""The minimum node value must be 0.""")

            if not max(node1_list + node2_list) == \
                    (len(set(node1_list + node2_list)) - 1):
                raise Exception("""All nodes must appear at least once in"""
                                + """ distances.""")

            path_list = list(zip(node1_list, node2_list))

        self.coords = coords
        self.distances = distances
        self.path_list = path_list
        self.dist_list = dist_list
        self.prob_type = 'tsp'

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation. Each integer between 0 and
            (len(state) - 1), inclusive must appear exactly once in the array.

        Returns
        -------
        fitness: float
            Value of fitness function. Returns :code:`np.inf` if travel between
            two consecutive nodes on the tour is not possible.
        """

        if self.is_coords and len(state) != len(self.coords):
            raise Exception("""state must have the same length as coords.""")

        if not len(state) == len(set(state)):
            raise Exception("""Each node must appear exactly once in state.""")

        if min(state) < 0:
            raise Exception("""All elements of state must be non-negative"""
                            + """ integers.""")

        if max(state) >= len(state):
            raise Exception("""All elements of state must be less than"""
                            + """ len(state).""")

        return self.calculate_fitness(state)

    def calculate_fitness(self, state):
        fitness = 0
        # Calculate length of each leg of journey
        for i in range(len(state) - 1):
            node1 = state[i]
            node2 = state[i + 1]

            if self.is_coords:
                fitness += np.linalg.norm(np.array(self.coords[node1])
                                          - np.array(self.coords[node2]))
            else:
                path = (min(node1, node2), max(node1, node2))

                if path in self.path_list:
                    fitness += self.dist_list[self.path_list.index(path)]
                else:
                    fitness += np.inf
        # Calculate length of final leg
        node1 = state[-1]
        node2 = state[0]
        if self.is_coords:
            fitness += np.linalg.norm(np.array(self.coords[node1])
                                      - np.array(self.coords[node2]))
        else:
            path = (min(node1, node2), max(node1, node2))

            if path in self.path_list:
                fitness += self.dist_list[self.path_list.index(path)]
            else:
                fitness += np.inf
        return fitness

    def get_prob_type(self):
        """ Return the problem type.

        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp'
            or 'either'.
        """
        return self.prob_type
