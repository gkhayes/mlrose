""" Classes for defining fitness functions.

    Author: Genevieve Hayes
    License: 3-clause BSD license.
"""
import numpy as np


class OneMax:
    """Fitness function for One Max optimization problem."""

    def __init__(self):
        """Initialize OneMax object.

        Args:
        None

        Returns:
        None
        """
        self.prob_type = 'either'

    def evaluate(self, state):
        """Evaluate the fitness of a state vector

        Args:
        state: array. State array for evaluation.

        Returns:
        fitness: float. Value of fitness function.
        """

        fitness = sum(state)
        return fitness

    def get_prob_type(self):
        """ Return the problem type

        Args:
        None

        Returns:
        self.prob_type: string. Specifies problem type as 'discrete',
        'continuous' or 'either'
        """
        return self.prob_type


class FlipFlop:
    """Fitness function for Flip Flop optimization problem."""

    def __init__(self):
        """Initialize FlipFlop object.

        Args:
        None

        Returns:
        None
        """
        self.prob_type = 'discrete'

    def evaluate(self, state):
        """Evaluate the fitness of a state vector

        Args:
        state: array. State array for evaluation.

        Returns:
        fitness: float. Value of fitness function.
        """
        fitness = 0

        for i in range(1, len(state)):
            if state[i] != state[i - 1]:
                fitness += 1

        return fitness

    def get_prob_type(self):
        """ Return the problem type

        Args:
        None

        Returns:
        self.prob_type: string. Specifies problem type as 'discrete',
        'continuous' or 'either'
        """
        return self.prob_type


def head(_b, _x):
    """Determine the number of leading b's in vector x

    Args:
    b: int. Integer for counting at head of vector.
    x: array. Vector of integers.

    Returns:
    head: int. Number of leading b's in x.
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
    """Determine the number of trailing b's in vector x

    Args:
    b: int. Integer for counting at tail of vector.
    x: array. Vector of integers.

    Returns:
    tail: int. Number of trailing b's in x.
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
    """Determine the length of the maximum run of b's in vector x

    Args:
    b: int. Integer for counting.
    x: array. Vector of integers.

    Returns:
    max: int. Length of maximum run of b's
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
    """Fitness function for Four Peaks optimization problem."""

    def __init__(self, t_pct=0.1):
        """Initialize FourPeaks object.

        Args:
        t_pct: float. Threshold parameter (T) for four peaks fitness function.

        Returns:
        None
        """
        self.t_pct = t_pct
        self.prob_type = 'discrete'

        if (self.t_pct < 0) or (self.t_pct > 1):
            raise Exception("""t_pct must be between 0 and 1.""")

    def evaluate(self, state):
        """Evaluate the fitness of a state

        Args:
        state: array. State array for evaluation.

        Returns:
        fitness: float. Value of fitness function.
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
        """ Return the problem type

        Args:
        None

        Returns:
        self.prob_type: string. Specifies problem type as 'discrete',
        'continuous' or 'either'
        """
        return self.prob_type


class SixPeaks:
    """Fitness function for Six Peaks optimization problem."""

    def __init__(self, t_pct=0.1):
        """Initialize SixPeaks object.

        Args:
        t_pct: float. Threshold parameter (T) for six peaks fitness function.

        Returns:
        None
        """
        self.t_pct = t_pct
        self.prob_type = 'discrete'

        if (self.t_pct < 0) or (self.t_pct > 1):
            raise Exception("""t_pct must be between 0 and 1.""")

    def evaluate(self, state):
        """Evaluate the fitness of a state

        Args:
        state: array. State array for evaluation.

        Returns:
        fitness: float. Value of fitness function.
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
        """ Return the problem type

        Args:
        None

        Returns:
        self.prob_type: string. Specifies problem type as 'discrete',
        'continuous' or 'either'
        """
        return self.prob_type


class ContinuousPeaks:
    """Fitness function for Continuous Peaks optimization problem."""

    def __init__(self, t_pct=0.1):
        """Initialize ContinuousPeaks object.

        Args:
        t_pct: float. Threshold parameter (T) for continuous peaks
        fitness function.

        Returns:
        None
        """
        self.t_pct = t_pct
        self.prob_type = 'discrete'

        if (self.t_pct < 0) or (self.t_pct > 1):
            raise Exception("""t_pct must be between 0 and 1.""")

    def evaluate(self, state):
        """Evaluate the fitness of a state

        Args:
        state: array. State array for evaluation.

        Returns:
        fitness: float. Value of fitness function.
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
        """ Return the problem type

        Args:
        None

        Returns:
        self.prob_type: string. Specifies problem type as 'discrete',
        'continuous' or 'either'
        """
        return self.prob_type


class Knapsack:
    """Fitness function for Knapsack optimization problem."""

    def __init__(self, weights, values, max_weight_pct=0.35):
        """Initialize Knapsack object.

        Args:
        weights: array. Array of weights for each of the possible items.
        values: array. Array of values for each of the possible items.
        Must be same length as weights array.
        max_weight_pct: float. Parameter used to set maximum capacity
        of knapsack (W) as a percentage of the total of the weights array
        (W = max_weight_pct*total_weight)

        Returns:
        None
        """
        self.weights = weights
        self.values = values
        self._w = np.ceil(sum(self.weights)*max_weight_pct)
        self.prob_type = 'discrete'

        if len(self.weights) != len(self.values):
            raise Exception("""The weights array and values array must be"""
                            + """ the same size.""")

        if min(self.weights) <= 0:
            raise Exception("""All weights must be greater than 0.""")

        if min(self.values) <= 0:
            raise Exception("""All values must be greater than 0.""")

        if (max_weight_pct <= 0) or (max_weight_pct > 1):
            raise Exception("""max_weight_pct must be between 0 and 1.""")

    def evaluate(self, state):
        """Evaluate the fitness of a state

        Args:
        state: array. State array for evaluation. Must be same length as
        weights and values arrays.

        Returns:
        fitness: float. Value of fitness function.
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
        """ Return the problem type

        Args:
        None

        Returns:
        self.prob_type: string. Specifies problem type as 'discrete',
        'continuous' or 'either'
        """
        return self.prob_type


class TravellingSales:
    """Fitness function for Travelling Salesman optimization problem."""

    def __init__(self, coords=None, distances=None):
        """Initialize TravellingSales object.

        Args:
        coords: list of pairs. Ordered list of the (x, y) co-ordinates of all
        nodes. This assumes that travel between all pairs of nodes is 
        possible. If this is not the case, then use distances instead.
        distances: list of triples. List giving the distances, d, between all 
        pairs of nodes, u and v, for which travel is possible, with each 
        list item in the form (u, v, d). Order of the nodes does not matter, 
        so (u, v, d) and (v, u, d) are considered to be the same. If a pair is
        missing from the list, it is assumed that travel between the two 
        nodes is not possible. This argument is ignored if coords is not
        None.

        Returns:
        None
        """
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
            distances = list({tuple(sorted(dist[0:2]) + [dist[2]]) \
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
        """Evaluate the fitness of a state

        Args:
        state: array. State array for evaluation. Each integer between 0 and 
        (len(state) - 1) must appear exactly once in the array.

        Returns:
        fitness: float. Value of fitness function.
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
        
        fitness = 0
        
        # Calculate length of each leg of journey
        for i in range(len(state) - 1):
            node1 = state[i]
            node2 = state[i + 1]
            
            if self.is_coords:
                fitness += np.linalg.norm(np.array(self.coords[node1]) \
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
            fitness += np.linalg.norm(np.array(self.coords[node1]) \
                                      - np.array(self.coords[node2]))
        else:
            path = (min(node1, node2), max(node1, node2))
            
            if path in self.path_list:
                fitness += self.dist_list[self.path_list.index(path)]
            else:
                fitness += np.inf

        return fitness

    def get_prob_type(self):
        """ Return the problem type

        Args:
        None

        Returns:
        self.prob_type: string. Specifies problem type as 'discrete',
        'continuous' or 'either'
        """
        return self.prob_type


class Queens:
    """Fitness function for N-Queens optimization problem."""

    def __init__(self):
        """Initialize Queens object.

        Args:
        None

        Returns:
        None
        """
        self.prob_type = 'discrete'

    def evaluate(self, state):
        """Evaluate the fitness of a state vector

        Args:
        state: array. State array for evaluation.

        Returns:
        fitness: float. Value of fitness function.
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
        """ Return the problem type

        Args:
        None

        Returns:
        self.prob_type: string. Specifies problem type as 'discrete',
        'continuous' or 'either'
        """
        return self.prob_type


class MaxKColor:
    """Fitness function for max-k color optimization problem."""

    def __init__(self, edges):
        """Initialize MaxKColor object.

        Args:
        edges: list of pairs. List of all pairs of connected nodes. Order
        does not matter, so (a, b) and (b, a) are considered to be the same.

        Returns:
        None
        """
        # Remove any duplicates from list
        edges = list({tuple(sorted(edge)) for edge in edges})

        self.edges = edges
        self.prob_type = 'discrete'

    def evaluate(self, state):
        """Evaluate the fitness of a state

        Args:
        state: array. State array for evaluation. Must contain the same number
        of elements as the distances matrix

        Returns:
        fitness: float. Value of fitness function.
        """
        fitness = 0

        for i in range(len(self.edges)):
            # Check for adjacent nodes of the same color
            if state[self.edges[i][0]] == state[self.edges[i][1]]:
                fitness += 1

        return fitness

    def get_prob_type(self):
        """ Return the problem type

        Args:
        None

        Returns:
        self.prob_type: string. Specifies problem type as 'discrete',
        'continuous' or 'either'
        """
        return self.prob_type


class CustomFitness:
    """Class for generating your own fitness function."""

    def __init__(self, fitness_fn, problem_type='either', **kwargs):
        """Initialize CustomFitness object.

        Args:
        fitness_fn: function. Function for calculating fitness of a state
        problem_type: string. Specifies problem type as 'discrete',
        'continuous' or 'either'
        kwargs: dictionary. Additional arguments to be passed to fitness_fn

        Returns:
        None
        """
        self.fitness_fn = fitness_fn
        self.problem_type = problem_type
        self.kwargs = kwargs

    def evaluate(self, state):
        """Evaluate the fitness of a state

        Args:
        state: array. State array for evaluation. Must contain the same number
        of elements as the distances matrix

        Returns:
        fitness: float. Value of fitness function.
        """
        fitness = self.fitness_fn(state, **self.kwargs)
        return fitness

    def get_prob_type(self):
        """ Return the problem type

        Args:
        None

        Returns:
        self.prob_type: string. Specifies problem type as 'discrete',
        'continuous' or 'either'
        """
        return None  # 'return prob_type' does not exist as a variable name
