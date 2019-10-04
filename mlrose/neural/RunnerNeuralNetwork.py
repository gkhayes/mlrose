""" Classes for defining neural network weight optimization problems."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np
import mlrose.algorithms as alg
import mlrose.neural.activation as act

from sklearn.base import ClassifierMixin
from mlrose.neural._NNBase import _NNBase


class RunnerNN(_NNBase, ClassifierMixin):
    def __init__(self,
                 runner,
                 # grid-search
                 algorithm=alg.simulated_annealing,
                 activation=act.relu,
                 hidden_nodes=None,
                 max_iters=100,
                 max_attempts=10,
                 learning_rate=0.1,
                 bias=True,
                 early_stopping=False,
                 clip_max=1e+10,
                 # general
                 callback_function_=None,
                 seed_=0,
                 cv_=5,
                 **kwargs
                 ):
        super().__init__()

        self.runner = runner

        # nn specific properties
        #  (grid-search settable)
        self.hidden_nodes = hidden_nodes
        self.activation = activation
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.max_iters = max_iters
        self.max_attempts = max_attempts

        #  (non-grid-search settable)
        self.bias = bias
        self.early_stopping = early_stopping
        self.clip_max = clip_max
        self.callback_function = callback_function_
        self.seed = seed_
        self.cv = cv_

        # result properties
        self.fitness_fn = None
        self.problem = None
        self.fitted_weights = None
        self.output_activation = None
        self.predicted_probabilities = None
        self.node_list = None
        self.loss = None

        # handle extra args
        self._process_kwargs(kwargs)

        # general runner properties.
        """
                 # general
                 max_iters=100,
                 runner=None,
                 max_attempts=10,
                 random_state=None,
                 curve=False,    

# rhc
                 restarts=0,
                 # sa
                 schedule=GeomDecay(),
                 # ga/mimic
                 pop_size=200,
                 # ga
                 mutation_prob=0.1                     
        """

    def _process_kwargs(self, kwargs):
        pass

    def get_params(self, deep=True):
        # exclude any that end with an underscore
        raw = super().get_params(deep)
        out = {k: v for (k, v) in raw.items() if not k[-1] == '_'}
        return out

    def _build_experiment_params(self):
        return {}

    def fit(self, x_train, y_train=None, init_weights=None):

        x_train, y_train = self._format_x_y_data(x_train, y_train)

        node_list = _NNBase._build_node_list(X=x_train, y=y_train,
                                             hidden_nodes=self.hidden_nodes,
                                             bias=self.bias)

        fitness, problem = _NNBase._build_problem_and_fitness_function(X=x_train,
                                                                       y=y_train,
                                                                       node_list=node_list,
                                                                       activation=self.activation,
                                                                       learning_rate=self.learning_rate,
                                                                       bias=self.bias,
                                                                       clip_max=self.clip_max)
        self.fitness_fn = fitness
        self.problem = problem

        state_size = _NNBase._calculate_state_size(node_list)
        params = self._build_experiment_params()
        if self.algorithm is not None:
            run_stats_df, curves_df = self.runner.run_experiment_(algorithm=self.algorithm,
                                                                  save_data=False,
                                                                  **params)
            fitted_weights = None  # best state
            loss = None  # not sure

        else:  # Gradient descent case
            _, fitted_weights, loss = self._run_gd(init_weights=init_weights,
                                                   num_nodes=state_size,
                                                   problem=problem,
                                                   curve=self.runner.curve,
                                                   early_stopping=self.early_stopping,
                                                   max_attempts=self.max_attempts,
                                                   max_iters=self.max_iters)

        # Save fitted weights
        self.node_list = node_list
        self.fitted_weights = fitted_weights
        self.loss = loss
        self.output_activation = self.fitness_fn.get_output_activation()

        return self

    def predict(self, x_test):
        if not np.shape(x_test)[1] == (self.node_list[0] - self.bias):
            raise Exception("""The number of columns in X must equal %d"""
                            % ((self.node_list[0] - self.bias),))

        y_pred, pp = _NNBase._predict(X=x_test,
                                      fitted_weights=self.fitted_weights,
                                      node_list=self.node_list,
                                      input_activation=self.activation,
                                      output_activation=self.output_activation,
                                      bias=self.bias)

        self.predicted_probabilities = pp
        return y_pred
