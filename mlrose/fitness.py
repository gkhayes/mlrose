""" Classes for defining fitness functions."""

# Author: Genevieve Hayes
# License: BSD 3 clause

import numpy as np


class OneMax:
    """Fitness function for One Max optimization problem. Evaluates the
    fitness of an n-dimensional state vector
    :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]` as:

    .. math::
        Fitness(x) = \\sum_{i = 0}^{n-1}x_{i}

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose
        >>> import numpy as np
        >>> fitness = mlrose.OneMax()
        >>> state = np.array([0, 1, 0, 1, 1, 1, 1])
        >>> fitness.evaluate(state)
        5

    Note
    -----
    The One Max fitness function is suitable for use in either discrete or
    continuous-state optimization problems.
    """

    def __init__(self):

        self.prob_type = 'either'

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        fitness = np.sum(state)
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


class FlipFlop:
    """Fitness function for Flip Flop optimization problem. Evaluates the
    fitness of a state vector :math:`x` as the total number of pairs of
    consecutive elements of :math:`x`, (:math:`x_{i}` and :math:`x_{i+1}`)
    where :math:`x_{i} \\neq x_{i+1}`.

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose
        >>> import numpy as np
        >>> fitness = mlrose.FlipFlop()
        >>> state = np.array([0, 1, 0, 1, 1, 1, 1])
        >>> fitness.evaluate(state)
        3

    Note
    ----
    The Flip Flop fitness function is suitable for use in discrete-state
    optimization problems *only*.
    """

    def __init__(self):

        self.prob_type = 'discrete'

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        fitness = 0

        for i in range(1, len(state)):
            if state[i] != state[i - 1]:
                fitness += 1

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


def head(_b, _x):
    """Determine the number of leading b's in vector x.

    Parameters
    ----------
    b: int
        Integer for counting at head of vector.
    x: array
        Vector of integers.

    Returns
    -------
    head: int
        Number of leading b's in x.
    """

    # Initialize counter
    _head = 0

    # Iterate through values in vector
    for i in _x:
        if i == _b:
            _head += 1
        else:
            break

    return _head


def tail(_b, _x):
    """Determine the number of trailing b's in vector x.

    Parameters
    ----------
    b: int
        Integer for counting at tail of vector.

    x: array
        Vector of integers.

    Returns
    -------
    tail: int
        Number of trailing b's in x.
    """

    # Initialize counter
    _tail = 0

    # Iterate backwards through values in vector
    for i in range(len(_x)):
        if _x[len(_x) - i - 1] == _b:
            _tail += 1
        else:
            break

    return _tail


def max_run(_b, _x):
    """Determine the length of the maximum run of b's in vector x.

    Parameters
    ----------
    b: int
        Integer for counting.

    x: array
        Vector of integers.

    Returns
    -------
    max: int
        Length of maximum run of b's.
    """
    # Initialize counter
    _max = 0
    run = 0

    # Iterate through values in vector
    for i in _x:
        if i == _b:
            run += 1
        else:
            if run > _max:
                _max = run

            run = 0

    if (_x[-1] == _b) and (run > _max):
        _max = run

    return _max


class FourPeaks:
    """Fitness function for Four Peaks optimization problem. Evaluates the
    fitness of an n-dimensional state vector :math:`x`, given parameter T, as:

    .. math::
        Fitness(x, T) = \\max(tail(0, x), head(1, x)) + R(x, T)

    where:

    * :math:`tail(b, x)` is the number of trailing b's in :math:`x`;
    * :math:`head(b, x)` is the number of leading b's in :math:`x`;
    * :math:`R(x, T) = n`, if :math:`tail(0, x) > T` and
      :math:`head(1, x) > T`; and
    * :math:`R(x, T) = 0`, otherwise.

    Parameters
    ----------
    t_pct: float, default: 0.1
        Threshold parameter (T) for Four Peaks fitness function, expressed as
        a percentage of the state space dimension, n (i.e.
        :math:`T = t_{pct} \\times n`).

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose
        >>> import numpy as np
        >>> fitness = mlrose.FourPeaks(t_pct=0.15)
        >>> state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        >>> fitness.evaluate(state)
        16

    References
    ----------
    De Bonet, J., C. Isbell, and P. Viola (1997). MIMIC: Finding Optima by
    Estimating Probability Densities. In *Advances in Neural Information
    Processing Systems* (NIPS) 9, pp. 424–430.

    Note
    ----
    The Four Peaks fitness function is suitable for use in bit-string
    (discrete-state with :code:`max_val = 2`) optimization problems *only*.
    """

    def __init__(self, t_pct=0.1):

        self.t_pct = t_pct
        self.prob_type = 'discrete'

        if (self.t_pct < 0) or (self.t_pct > 1):
            raise Exception("""t_pct must be between 0 and 1.""")

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float.
            Value of fitness function.
        """
        _n = len(state)
        _t = np.ceil(self.t_pct*_n)

        # Calculate head and tail values
        tail_0 = tail(0, state)
        head_1 = head(1, state)

        # Calculate R(X, T)
        if (tail_0 > _t and head_1 > _t):
            _r = _n
        else:
            _r = 0

        # Evaluate function
        fitness = max(tail_0, head_1) + _r

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


class SixPeaks:
    """Fitness function for Six Peaks optimization problem. Evaluates the
    fitness of an n-dimensional state vector :math:`x`, given parameter T, as:

    .. math::
        Fitness(x, T) = \\max(tail(0, x), head(1, x)) + R(x, T)

    where:

    * :math:`tail(b, x)` is the number of trailing b's in :math:`x`;
    * :math:`head(b, x)` is the number of leading b's in :math:`x`;
    * :math:`R(x, T) = n`, if (:math:`tail(0, x) > T` and
      :math:`head(1, x) > T`) or (:math:`tail(1, x) > T` and
      :math:`head(0, x) > T`); and
    * :math:`R(x, T) = 0`, otherwise.

    Parameters
    ----------
    t_pct: float, default: 0.1
        Threshold parameter (T) for Six Peaks fitness function, expressed as
        a percentage of the state space dimension, n (i.e.
        :math:`T = t_{pct} \\times n`).

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose
        >>> import numpy as np
        >>> fitness = mlrose.SixPeaks(t_pct=0.15)
        >>> state = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1])
        >>> fitness.evaluate(state)
        12

    References
    ----------
    De Bonet, J., C. Isbell, and P. Viola (1997). MIMIC: Finding Optima by
    Estimating Probability Densities. In *Advances in Neural Information
    Processing Systems* (NIPS) 9, pp. 424–430.

    Note
    ----
    The Six Peaks fitness function is suitable for use in bit-string
    (discrete-state with :code:`max_val = 2`) optimization problems *only*.
    """

    def __init__(self, t_pct=0.1):

        self.t_pct = t_pct
        self.prob_type = 'discrete'

        if (self.t_pct < 0) or (self.t_pct > 1):
            raise Exception("""t_pct must be between 0 and 1.""")

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """
        _n = len(state)
        _t = np.ceil(self.t_pct*_n)

        # Calculate head and tail values
        head_0 = head(0, state)
        tail_0 = tail(0, state)
        head_1 = head(1, state)
        tail_1 = tail(1, state)

        # Calculate R(X, T)
        if (tail_0 > _t and head_1 > _t) or (tail_1 > _t and head_0 > _t):
            _r = _n
        else:
            _r = 0

        # Evaluate function
        fitness = max(tail_0, head_1) + _r

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


class ContinuousPeaks:
    """Fitness function for Continuous Peaks optimization problem. Evaluates
    the fitness of an n-dimensional state vector :math:`x`, given parameter T,
    as:

    .. math::
        Fitness(x, T) = \\max(max\\_run(0, x), max\\_run(1, x)) + R(x, T)

    where:

    * :math:`max\\_run(b, x)` is the length of the maximum run of b's
      in :math:`x`;
    * :math:`R(x, T) = n`, if (:math:`max\\_run(0, x) > T` and
      :math:`max\\_run(1, x) > T`); and
    * :math:`R(x, T) = 0`, otherwise.

    Parameters
    ----------
    t_pct: float, default: 0.1
        Threshold parameter (T) for Continuous Peaks fitness function,
        expressed as a percentage of the state space dimension, n (i.e.
        :math:`T = t_{pct} \\times n`).

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose
        >>> import numpy as np
        >>> fitness = mlrose.ContinuousPeaks(t_pct=0.15)
        >>> state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1])
        >>> fitness.evaluate(state)
        17

    Note
    ----
    The Continuous Peaks fitness function is suitable for use in bit-string
    (discrete-state with :code:`max_val = 2`) optimization problems *only*.
    """

    def __init__(self, t_pct=0.1):

        self.t_pct = t_pct
        self.prob_type = 'discrete'

        if (self.t_pct < 0) or (self.t_pct > 1):
            raise Exception("""t_pct must be between 0 and 1.""")

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """
        _n = len(state)
        _t = np.ceil(self.t_pct*_n)

        # Calculate length of maximum runs of 0's and 1's
        max_0 = max_run(0, state)
        max_1 = max_run(1, state)

        # Calculate R(X, T)
        if (max_0 > _t and max_1 > _t):
            _r = _n
        else:
            _r = 0

        # Evaluate function
        fitness = max(max_0, max_1) + _r

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


class Knapsack:
    """Fitness function for Knapsack optimization problem. Given a set of n
    items, where item i has known weight :math:`w_{i}` and known value
    :math:`v_{i}`; and maximum knapsack capacity, :math:`W`, the Knapsack
    fitness function evaluates the fitness of a state vector
    :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]` as:

    .. math::
        Fitness(x) = \\sum_{i = 0}^{n-1}v_{i}x_{i}, \\text{ if}
        \\sum_{i = 0}^{n-1}w_{i}x_{i} \\leq W, \\text{ and 0, otherwise,}

    where :math:`x_{i}` denotes the number of copies of item i included in the
    knapsack.

    Parameters
    ----------
    weights: list
        List of weights for each of the n items.

    values: list
        List of values for each of the n items.

    max_weight_pct: float, default: 0.35
        Parameter used to set maximum capacity of knapsack (W) as a percentage
        of the total of the weights list
        (:math:`W =` max_weight_pct :math:`\\times` total_weight).

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose
        >>> import numpy as np
        >>> weights = [10, 5, 2, 8, 15]
        >>> values = [1, 2, 3, 4, 5]
        >>> max_weight_pct = 0.6
        >>> fitness = mlrose.Knapsack(weights, values, max_weight_pct)
        >>> state = np.array([1, 0, 2, 1, 0])
        >>> fitness.evaluate(state)
        11

    Note
    ----
    The Knapsack fitness function is suitable for use in discrete-state
    optimization problems *only*.
    """

    def __init__(self, weights, values, max_weight_pct=0.35):

        self.weights = weights
        self.values = values
        self._w = np.ceil(np.sum(self.weights)*max_weight_pct)
        self.prob_type = 'discrete'

        if len(self.weights) != len(self.values):
            raise Exception("""The weights array and values array must be"""
                            + """ the same size.""")

        if min(self.weights) <= 0:
            raise Exception("""All weights must be greater than 0.""")

        if min(self.values) <= 0:
            raise Exception("""All values must be greater than 0.""")

        if max_weight_pct <= 0:
            raise Exception("""max_weight_pct must be greater than 0.""")

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation. Must be the same length as the weights
            and values arrays.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        if len(state) != len(self.weights):
            raise Exception("""The state array must be the same size as the"""
                            + """ weight and values arrays.""")

        # Calculate total weight and value of knapsack
        total_weight = np.sum(state*self.weights)
        total_value = np.sum(state*self.values)

        # Allow for weight constraint
        if total_weight <= self._w:
            fitness = total_value
        else:
            fitness = 0

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


class Queens:
    """Fitness function for N-Queens optimization problem. Evaluates the
    fitness of an n-dimensional state vector
    :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]`, where :math:`x_{i}`
    represents the row position (between 0 and n-1, inclusive) of the 'queen'
    in column i, as the number of pairs of attacking queens.

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose
        >>> import numpy as np
        >>> fitness = mlrose.Queens()
        >>> state = np.array([1, 4, 1, 3, 5, 5, 2, 7])
        >>> fitness.evaluate(state)
        6

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.

    Note
    ----
    The Queens fitness function is suitable for use in discrete-state
    optimization problems *only*.
    """

    def __init__(self):

        self.prob_type = 'discrete'

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        fitness = 0

        for i in range(len(state) - 1):
            for j in range(i + 1, len(state)):
                # Check for horizontal attacks
                if state[j] == state[i]:
                    fitness += 1

                # Check for diagonal-up attacks
                elif state[j] == state[i] + (j - i):
                    fitness += 1

                # Check for diagonal-down attacks
                elif state[j] == state[i] - (j - i):
                    fitness += 1

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


class MaxKColor:
    """Fitness function for Max-k color optimization problem. Evaluates the
    fitness of an n-dimensional state vector
    :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]`, where :math:`x_{i}`
    represents the color of node i, as the number of pairs of adjacent nodes
    of the same color.

    Parameters
    ----------
    edges: list of pairs
        List of all pairs of connected nodes. Order does not matter, so (a, b)
        and (b, a) are considered to be the same.

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose
        >>> import numpy as np
        >>> edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        >>> fitness = mlrose.MaxKColor(edges)
        >>> state = np.array([0, 1, 0, 1, 1])
        >>> fitness.evaluate(state)
        3

    Note
    ----
    The MaxKColor fitness function is suitable for use in discrete-state
    optimization problems *only*.
    """

    def __init__(self, edges):

        # Remove any duplicates from list
        edges = list({tuple(sorted(edge)) for edge in edges})

        self.edges = edges
        self.prob_type = 'discrete'

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        fitness = 0

        for i in range(len(self.edges)):
            # Check for adjacent nodes of the same color
            if state[self.edges[i][0]] == state[self.edges[i][1]]:
                fitness += 1

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


class CustomFitness:
    """Class for generating your own fitness function.

    Parameters
    ----------
    fitness_fn: callable
        Function for calculating fitness of a state with the signature
        :code:`fitness_fn(state, **kwargs)`.

    problem_type: string, default: 'either'
        Specifies problem type as 'discrete', 'continuous', 'tsp' or 'either'
        (denoting either discrete or continuous).

    kwargs: additional arguments
        Additional parameters to be passed to the fitness function.

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose
        >>> import numpy as np
        >>> def cust_fn(state, c): return c*np.sum(state)
        >>> kwargs = {'c': 10}
        >>> fitness = mlrose.CustomFitness(cust_fn, **kwargs)
        >>> state = np.array([1, 2, 3, 4, 5])
        >>> fitness.evaluate(state)
        150
    """

    def __init__(self, fitness_fn, problem_type='either', **kwargs):

        if problem_type not in ['discrete', 'continuous', 'tsp', 'either']:
            raise Exception("""problem_type does not exist.""")
        self.fitness_fn = fitness_fn
        self.problem_type = problem_type
        self.kwargs = kwargs

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        fitness = self.fitness_fn(state, **self.kwargs)
        return fitness

    def get_prob_type(self):
        """ Return the problem type.

        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp'
            or 'either'.
        """
        return self.problem_type
