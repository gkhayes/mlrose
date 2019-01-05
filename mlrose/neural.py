""" Classes for defining neural network weight optimization problems."""

# Author: Genevieve Hayes
# License: BSD 3 clause

import numpy as np
from sklearn.metrics import mean_squared_error, log_loss
from .activation import identity, relu, sigmoid, softmax, tanh
from .algorithms import random_hill_climb, simulated_annealing, genetic_alg
from .opt_probs import ContinuousOpt
from .decay import GeomDecay


def flatten_weights(weights):
    """Flatten list of weights arrays into a 1D array.

    Parameters
    ----------
    weights: list of arrays
        List of 2D arrays for flattening.

    Returns
    -------
    flat_weights: array
        1D weights array.
    """
    flat_weights = []

    for i in range(len(weights)):
        flat_weights += list(weights[i].flatten())

    flat_weights = np.array(flat_weights)

    return flat_weights


def unflatten_weights(flat_weights, node_list):
    """Convert 1D weights array into list of 2D arrays.

    Parameters
    ----------
    flat_weights: array
        1D weights array.

    node_list: list
        List giving the number of nodes in each layer of the network,
        including the input and output layers.

    Returns
    -------
    weights: list of arrays
        List of 2D arrays created from flat_weights.
    """
    nodes = 0
    for i in range(len(node_list) - 1):
        nodes += node_list[i]*node_list[i + 1]

    if len(flat_weights) != nodes:
        raise Exception("""flat_weights must have length %d""" % (nodes,))

    weights = []
    start = 0

    for i in range(len(node_list) - 1):
        end = start + node_list[i]*node_list[i + 1]
        weights.append(np.reshape(flat_weights[start:end],
                                  [node_list[i], node_list[i+1]]))
        start = end

    return weights


def gradient_descent(problem, max_attempts=10, max_iters=np.inf,
                     init_state=None):
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

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes fitness function.

    best_fitness: float
        Value of fitness function at best state.
    """
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Initialize problem, time and attempts counter
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    attempts = 0
    iters = 0

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Update weights
        updates = flatten_weights(problem.calculate_updates())

        next_state = problem.update_state(updates)
        next_fitness = problem.eval_fitness(next_state)

        if next_fitness > problem.get_fitness():
            attempts = 0
        else:
            attempts += 1

        if next_fitness > problem.get_maximize()*best_fitness:
            best_fitness = problem.get_maximize()*next_fitness
            best_state = next_state

        problem.set_state(next_state)

    return best_state, best_fitness


class NetworkWeights:
    """Fitness function for neural network weights optimization problem.

    Parameters
    ----------
    X: array
        Numpy array containing feature dataset with each row representing a
        single observation.

    y: array
        Numpy array containing true values of data labels.
        Length must be same as length of X.

    node_list: list of ints
        Number of nodes in each layer, including the input and output layers.

    activation: callable
        Activation function for each of the hidden layers with the signature
        :code:`activation(x, deriv)`, where setting deriv is a boolean that
        determines whether to return the activation function or its derivative.

    bias: bool, default: True
        Whether a bias term is included in the network.

    is_classifer: bool, default: True
        Whether the network is for classification orregression. Set True for
        classification and False for regression.
    """

    def __init__(self, X, y, node_list, activation, bias=True,
                 is_classifier=True, learning_rate=0.1):

        # Make sure y is an array and not a list
        y = np.array(y)

        # Convert y to 2D if necessary
        if len(np.shape(y)) == 1:
            y = np.reshape(y, [len(y), 1])

        # Verify X and y are the same length
        if not np.shape(X)[0] == np.shape(y)[0]:
            raise Exception("""The length of X and y must be equal.""")

        if len(node_list) < 2:
            raise Exception("""node_list must contain at least 2 elements.""")

        if not np.shape(X)[1] == (node_list[0] - bias):
            raise Exception("""The number of columns in X must equal %d"""
                            % ((node_list[0] - bias),))

        if not np.shape(y)[1] == node_list[-1]:
            raise Exception("""The number of columns in y must equal %d"""
                            % (node_list[-1],))

        if not isinstance(bias, bool):
            raise Exception("""bias must be True or False.""")

        if not isinstance(is_classifier, bool):
            raise Exception("""is_classifier must be True or False.""")

        if learning_rate <= 0:
            raise Exception("""learning_rate must be greater than 0.""")

        self.X = X
        self.y_true = y
        self.node_list = node_list
        self.activation = activation
        self.bias = bias
        self.is_classifier = is_classifier
        self.lr = learning_rate

        # Determine appropriate loss function and output activation function
        if self.is_classifier:
            self.loss = log_loss

            if np.shape(self.y_true)[1] == 1:
                self.output_activation = sigmoid
            else:
                self.output_activation = softmax
        else:
            self.loss = mean_squared_error
            self.output_activation = identity

        self.inputs_list = []
        self.y_pred = y
        self.weights = []
        self.prob_type = 'continuous'

        nodes = 0
        for i in range(len(node_list) - 1):
            nodes += node_list[i]*node_list[i + 1]

        self.nodes = nodes

    def evaluate(self, state):
        """Evaluate the fitness of a state.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """
        if not len(state) == self.nodes:
            raise Exception("""state must have length %d""" % (self.nodes,))

        self.inputs_list = []
        self.weights = unflatten_weights(state, self.node_list)

        # Add bias column to inputs matrix, if required
        if self.bias:
            ones = np.ones([np.shape(self.X)[0], 1])
            inputs = np.hstack((self.X, ones))

        else:
            inputs = self.X

        # Pass data through network
        for i in range(len(self.weights)):
            # Multiple inputs by weights
            outputs = np.dot(inputs, self.weights[i])
            self.inputs_list.append(inputs)

            # Transform outputs to get inputs for next layer (or final preds)
            if i < len(self.weights) - 1:
                inputs = self.activation(outputs)
            else:
                self.y_pred = self.output_activation(outputs)

        # Evaluate loss function
        fitness = self.loss(self.y_true, self.y_pred)

        return fitness

    def get_output_activation(self):
        """ Return the activation function for the output layer.

        Returns
        -------
        self.output_activation: callable
            Activation function for the output layer.
        """
        return self.output_activation

    def get_prob_type(self):
        """ Return the problem type.

        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp', or
            'either'.
        """
        return self.prob_type

    def calculate_updates(self):
        """Calculate gradient descent updates.

        Returns
        -------
        updates_list: list
            List of back propagation weight updates.
        """
        delta_list = []
        updates_list = []

        # Work backwards from final layer
        for i in range(len(self.inputs_list)-1, -1, -1):
            # Final layer
            if i == len(self.inputs_list)-1:
                delta = (self.y_pred - self.y_true)
            # Hidden layers
            else:
                dot = np.dot(delta_list[-1], np.transpose(self.weights[i+1]))
                activation = self.activation(self.inputs_list[i+1], deriv=True)
                delta = dot * activation

            delta_list.append(delta)

            # Calculate updates
            updates = -1.0*self.lr*np.dot(np.transpose(self.inputs_list[i]),
                                          delta)

            updates_list.append(updates)

        # Reverse order of updates list
        updates_list = updates_list[::-1]

        return updates_list


class NeuralNetwork:
    """Class for defining neural network weights optimization problem.

    Parameters
    ----------
    hidden_nodes: list of ints
        List giving the number of nodes in each hidden layer.

    activation: string, default: 'relu'
        Activation function for each of the hidden layers. Must be one of:
        'identity', 'relu', 'sigmoid' or 'tanh'.

    algorithm: string, default: 'random_hill_climb'
        Algorithm used to find optimal network weights. Must be one
        of:'random_hill_climb', 'simulated_annealing', 'genetic_alg' or
        'gradient_descent'.

    max_iters: int, default: 100
        Maximum number of iterations used to fit the weights.

    bias: bool, default: True
        Whether to include a bias term.

    is_classifer: bool, default: True
        Whether the network is for classification or regression. Set
        :code:`True` for classification and :code:`False` for regression.

    learning_rate: float, default: 0.1
        Learning rate for gradient descent or step size for randomized
        optimization algorithms.

    early_stopping: bool, default: False
        Whether to terminate algorithm early if the loss is not improving.
        If :code:`True`, then stop after max_attempts iters with no
        improvement.

    clip_max: float, default: 1e+10
        Used to limit weights to the range [-1*clip_max, clip_max].

    schedule: schedule object, default = mlrose.GeomDecay()
        Schedule used to determine the value of the temperature parameter.
        Only required if :code:`algorithm = 'simulated_annealing'`.

    pop_size: int, default: 200
        Size of population. Only required if :code:`algorithm = 'genetic_alg'`.

    mutation_prob: float, default: 0.1
        Probability of a mutation at each element of the state vector during
        reproduction, expressed as a value between 0 and 1. Only required if
        :code:`algorithm = 'genetic_alg'`.

    max_attempts: int, default: 10
        Maximum number of attempts to find a better state. Only required if
        :code:`early_stopping = True`.

    Attributes
    ----------
    fitted_weights: array
        Numpy array giving the fitted weights when :code:`fit` is performed.

    loss: float
        Value of loss function for fitted weights when :code:`fit` is
        performed.

    predicted_probs: array
        Numpy array giving the predicted probabilities for each class when
        :code:`predict` is performed for multi-class classification data; or
        the predicted probability for class 1 when :code:`predict` is performed
        for binary classification data.
    """

    def __init__(self, hidden_nodes, activation='relu',
                 algorithm='random_hill_climb', max_iters=100, bias=True,
                 is_classifier=True, learning_rate=0.1, early_stopping=False,
                 clip_max=1e+10, schedule=GeomDecay(), pop_size=200,
                 mutation_prob=0.1, max_attempts=10):

        if (not isinstance(max_iters, int) and max_iters != np.inf
                and not max_iters.is_integer()) or (max_iters < 0):
            raise Exception("""max_iters must be a positive integer.""")

        if not isinstance(bias, bool):
            raise Exception("""bias must be True or False.""")

        if not isinstance(is_classifier, bool):
            raise Exception("""is_classifier must be True or False.""")

        if learning_rate <= 0:
            raise Exception("""learning_rate must be greater than 0.""")

        if not isinstance(early_stopping, bool):
            raise Exception("""early_stopping must be True or False.""")

        if clip_max <= 0:
            raise Exception("""clip_max must be greater than 0.""")

        if (not isinstance(max_attempts, int)
                and not max_attempts.is_integer()) or (max_attempts < 0):
            raise Exception("""max_attempts must be a positive integer.""")

        if pop_size < 0:
            raise Exception("""pop_size must be a positive integer.""")
        elif not isinstance(pop_size, int):
            if pop_size.is_integer():
                pop_size = int(pop_size)
            else:
                raise Exception("""pop_size must be a positive integer.""")

        if (mutation_prob < 0) or (mutation_prob > 1):
            raise Exception("""mutation_prob must be between 0 and 1.""")

        self.hidden_nodes = hidden_nodes
        self.max_iters = max_iters
        self.bias = bias
        self.is_classifier = is_classifier
        self.lr = learning_rate
        self.early_stopping = early_stopping
        self.clip_max = clip_max
        self.schedule = schedule
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob

        activation_dict = {'identity': identity, 'relu': relu,
                           'sigmoid': sigmoid, 'tanh': tanh}
        if activation in activation_dict.keys():
            self.activation = activation_dict[activation]
        else:
            raise Exception("""Activation function must be one of: 'identity',
            'relu', 'sigmoid' or 'tanh'.""")

        if algorithm in ['random_hill_climb', 'simulated_annealing',
                         'genetic_alg', 'gradient_descent']:
            self.algorithm = algorithm
        else:
            raise Exception("""Algorithm must be one of: 'random_hill_climb',
            'simulated_annealing', 'genetic_alg', 'gradient_descent'.""")

        if self.early_stopping:
            self.max_attempts = max_attempts
        else:
            self.max_attempts = self.max_iters

        self.node_list = []
        self.fitted_weights = []
        self.loss = np.inf
        self.output_activation = None
        self.predicted_probs = []

    def fit(self, X, y, init_weights=None):
        """Fit neural network to data.

        Parameters
        ----------
        X: array
            Numpy array containing feature dataset with each row
            representing a single observation.

        y: array
            Numpy array containing data labels. Length must be same as
            length of X.

        init_state: array, default: None
            Numpy array containing starting weights for algorithm.
            If :code:`None`, then a random state is used.
        """
        # Make sure y is an array and not a list
        y = np.array(y)

        # Convert y to 2D if necessary
        if len(np.shape(y)) == 1:
            y = np.reshape(y, [len(y), 1])

        # Verify X and y are the same length
        if not np.shape(X)[0] == np.shape(y)[0]:
            raise Exception('The length of X and y must be equal.')

        # Determine number of nodes in each layer
        input_nodes = np.shape(X)[1] + self.bias
        output_nodes = np.shape(y)[1]
        node_list = [input_nodes] + self.hidden_nodes + [output_nodes]

        num_nodes = 0

        for i in range(len(node_list) - 1):
            num_nodes += node_list[i]*node_list[i+1]

        if init_weights is not None and len(init_weights) != num_nodes:
            raise Exception("""init_weights must be None or have length %d"""
                            % (num_nodes,))

        # Initialize optimization problem
        fitness = NetworkWeights(X, y, node_list, self.activation, self.bias,
                                 self.is_classifier, learning_rate=self.lr)

        problem = ContinuousOpt(num_nodes, fitness, maximize=False,
                                min_val=-1*self.clip_max,
                                max_val=self.clip_max, step=self.lr)

        if self.algorithm == 'random_hill_climb':
            if init_weights is None:
                init_weights = np.random.uniform(-1, 1, num_nodes)

            fitted_weights, loss = random_hill_climb(
                problem,
                max_attempts=self.max_attempts, max_iters=self.max_iters,
                restarts=0, init_state=init_weights)

        elif self.algorithm == 'simulated_annealing':
            if init_weights is None:
                init_weights = np.random.uniform(-1, 1, num_nodes)
            fitted_weights, loss = simulated_annealing(
                problem,
                schedule=self.schedule, max_attempts=self.max_attempts,
                max_iters=self.max_iters, init_state=init_weights)

        elif self.algorithm == 'genetic_alg':
            fitted_weights, loss = genetic_alg(
                problem,
                pop_size=self.pop_size, mutation_prob=self.mutation_prob,
                max_attempts=self.max_attempts, max_iters=self.max_iters)

        else:  # Gradient descent case
            if init_weights is None:
                init_weights = np.random.uniform(-1, 1, num_nodes)
            fitted_weights, loss = gradient_descent(
                problem,
                max_attempts=self.max_attempts, max_iters=self.max_iters,
                init_state=init_weights)

        # Save fitted weights and node list
        self.node_list = node_list
        self.fitted_weights = fitted_weights
        self.loss = loss
        self.output_activation = fitness.get_output_activation()

    def predict(self, X):
        """Use model to predict data labels for given feature array.

        Parameters
        ----------
        X: array
            Numpy array containing feature dataset with each row
            representing a single observation.

        Returns
        -------
        y_pred: array
            Numpy array containing predicted data labels.
        """
        if not np.shape(X)[1] == (self.node_list[0] - self.bias):
            raise Exception("""The number of columns in X must equal %d"""
                            % ((self.node_list[0] - self.bias),))

        weights = unflatten_weights(self.fitted_weights, self.node_list)

        # Add bias column to inputs matrix, if required
        if self.bias:
            ones = np.ones([np.shape(X)[0], 1])
            inputs = np.hstack((X, ones))

        else:
            inputs = X

        # Pass data through network
        for i in range(len(weights)):
            # Multiple inputs by weights
            outputs = np.dot(inputs, weights[i])

            # Transform outputs to get inputs for next layer (or final preds)
            if i < len(weights) - 1:
                inputs = self.activation(outputs)
            else:
                y_pred = self.output_activation(outputs)

        # For classifier, convert predicted probabilities to 0-1 labels
        if self.is_classifier:
            self.predicted_probs = y_pred

            if self.node_list[-1] == 1:
                y_pred = np.round(y_pred).astype(int)
            else:
                zeros = np.zeros_like(y_pred)
                zeros[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)] = 1
                y_pred = zeros.astype(int)

        return y_pred


class LinearRegression(NeuralNetwork):
    """Class for defining linear regression weights optimization
    problem. Inherits :code:`fit` and :code:`predict` methods from
    :code:`NeuralNetwork()` class.

    Parameters
    ----------
    algorithm: string, default: 'random_hill_climb'
        Algorithm used to find optimal network weights. Must be one
        of:'random_hill_climb', 'simulated_annealing', 'genetic_alg' or
        'gradient_descent'.

    max_iters: int, default: 100
        Maximum number of iterations used to fit the weights.

    bias: bool, default: True
        Whether to include a bias term.

    learning_rate: float, default: 0.1
        Learning rate for gradient descent or step size for randomized
        optimization algorithms.

    early_stopping: bool, default: False
        Whether to terminate algorithm early if the loss is not improving.
        If :code:`True`, then stop after max_attempts iters with no
        improvement.

    clip_max: float, default: 1e+10
        Used to limit weights to the range [-1*clip_max, clip_max].

    schedule: schedule object, default = mlrose.GeomDecay()
        Schedule used to determine the value of the temperature parameter.
        Only required if :code:`algorithm = 'simulated_annealing'`.

    pop_size: int, default: 200
        Size of population. Only required if :code:`algorithm = 'genetic_alg'`.

    mutation_prob: float, default: 0.1
        Probability of a mutation at each element of the state vector during
        reproduction, expressed as a value between 0 and 1. Only required if
        :code:`algorithm = 'genetic_alg'`.

    max_attempts: int, default: 10
        Maximum number of attempts to find a better state. Only required if
        :code:`early_stopping = True`.

    Attributes
    ----------
    fitted_weights: array
        Numpy array giving the fitted weights when :code:`fit` is performed.

    loss: float
        Value of loss function for fitted weights when :code:`fit` is
        performed.
    """

    def __init__(self, algorithm='random_hill_climb', max_iters=100, bias=True,
                 learning_rate=0.1, early_stopping=False, clip_max=1e+10,
                 schedule=GeomDecay(), pop_size=200, mutation_prob=0.1,
                 max_attempts=10):

        NeuralNetwork.__init__(
            self, hidden_nodes=[], activation='identity',
            algorithm=algorithm, max_iters=max_iters, bias=bias,
            is_classifier=False, learning_rate=learning_rate,
            early_stopping=early_stopping, clip_max=clip_max,
            schedule=schedule, pop_size=pop_size, mutation_prob=mutation_prob,
            max_attempts=max_attempts)


class LogisticRegression(NeuralNetwork):
    """Class for defining logistic regression weights optimization
    problem. Inherits :code:`fit` and :code:`predict` methods from
    :code:`NeuralNetwork()` class.

    Parameters
    ----------
    algorithm: string, default: 'random_hill_climb'
        Algorithm used to find optimal network weights. Must be one
        of:'random_hill_climb', 'simulated_annealing', 'genetic_alg' or
        'gradient_descent'.

    max_iters: int, default: 100
        Maximum number of iterations used to fit the weights.

    bias: bool, default: True
        Whether to include a bias term.

    learning_rate: float, default: 0.1
        Learning rate for gradient descent or step size for randomized
        optimization algorithms.

    early_stopping: bool, default: False
        Whether to terminate algorithm early if the loss is not improving.
        If :code:`True`, then stop after max_attempts iters with no
        improvement.

    clip_max: float, default: 1e+10
        Used to limit weights to the range [-1*clip_max, clip_max].

    schedule: schedule object, default = mlrose.GeomDecay()
        Schedule used to determine the value of the temperature parameter.
        Only required if :code:`algorithm = 'simulated_annealing'`.

    pop_size: int, default: 200
        Size of population. Only required if :code:`algorithm = 'genetic_alg'`.

    mutation_prob: float, default: 0.1
        Probability of a mutation at each element of the state vector during
        reproduction, expressed as a value between 0 and 1. Only required if
        :code:`algorithm = 'genetic_alg'`.

    max_attempts: int, default: 10
        Maximum number of attempts to find a better state. Only required if
        :code:`early_stopping = True`.

    Attributes
    ----------
    fitted_weights: array
        Numpy array giving the fitted weights when :code:`fit` is performed.

    loss: float
        Value of loss function for fitted weights when :code:`fit` is
        performed.
    """

    def __init__(self, algorithm='random_hill_climb', max_iters=100, bias=True,
                 learning_rate=0.1, early_stopping=False, clip_max=1e+10,
                 schedule=GeomDecay(), pop_size=200, mutation_prob=0.1,
                 max_attempts=10):

        NeuralNetwork.__init__(
            self, hidden_nodes=[], activation='sigmoid',
            algorithm=algorithm, max_iters=max_iters, bias=bias,
            is_classifier=True, learning_rate=learning_rate,
            early_stopping=early_stopping, clip_max=clip_max,
            schedule=schedule, pop_size=pop_size, mutation_prob=mutation_prob,
            max_attempts=max_attempts)
