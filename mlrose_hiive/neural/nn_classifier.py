""" Classes for defining neural network weight optimization problems."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.neural._nn_base import _NNBase


class NNClassifier(_NNBase):

    def __init__(self,
                 runner,
                 algorithm=None,
                 # grid-search
                 activation=None,
                 hidden_layer_sizes=None,
                 max_iters=100,
                 max_attempts=10,
                 learning_rate_init=0.1,
                 bias=True,
                 early_stopping=False,
                 clip_max=1e+10,
                 seed=None,
                 **kwargs
                 ):
        super().__init__()

        self.runner = runner
        self.grid_search_parameters = runner.grid_search_parameters

        # nn specific properties
        #  (grid-search settable)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_iters = max_iters
        self.max_attempts = max_attempts

        #  (non-grid-search settable)
        self.bias = bias
        self.early_stopping = early_stopping
        self.clip_max = clip_max
        self.algorithm = algorithm

        # result properties
        self.fitness_fn = None
        self.problem = None
        self.fitted_weights = None
        self.output_activation = None
        self.predicted_probabilities = None
        self.node_list = None
        self.node_count = None
        self.loss = None
        self.fit_started_ = False
        self.seed = seed

        # extra parameters
        self.kwargs = kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                continue
            self.__setattr__(k, v)

    def get_params(self, deep=True):
        out = super().get_params(deep)
        # exclude any that end with an underscore
        out = {k: v for (k, v) in out.items() if not k[-1] == '_'}
        ap = {k: self.__dict__[k] if k in self.__dict__ else None for k in self.grid_search_parameters if k not in out}
        out.update(ap)
        return out

    def _get_nodes(self, x_train, y_train):
        return _NNBase._build_node_list(X=x_train, y=y_train,
                                        hidden_nodes=self.hidden_layer_sizes,
                                        bias=self.bias)

    def fit(self, x_train, y_train=None, init_weights=None):

        x_train, y_train = self._format_x_y_data(x_train, y_train)
        self.node_list = self._get_nodes(x_train, y_train)
        self.node_count = _NNBase._calculate_state_size(self.node_list)
        fitness, problem = _NNBase._build_problem_and_fitness_function(X=x_train,
                                                                       y=y_train,
                                                                       node_list=self.node_list,
                                                                       activation=self.activation,
                                                                       learning_rate=self.learning_rate_init,
                                                                       bias=self.bias,
                                                                       clip_max=self.clip_max)
        self.fitness_fn = fitness
        self.problem = problem

        # check for early abort.
        if self.runner.has_aborted() or self.runner.replay_mode():
            self.fitted_weights = np.array([np.NaN] * self.node_count)
            self.loss = np.NaN
            self.output_activation = self.fitness_fn.get_output_activation()
            return self

        if self.algorithm is not None:
            # self._perform_grid_search()
            params = {k: self.__getattribute__(k) for k in self.kwargs}
            if init_weights is None:
                np.random.seed(self.seed)
                init_weights = np.random.uniform(-1, 1, self.node_count)
            params['init_state'] = init_weights
            total_args = {
                'algorithm': self.algorithm,
                'activation': self.activation,
                'bias': self.bias,
                'early_stopping': self.early_stopping,
                'clip_max': self.clip_max,
                'hidden_layer_sizes': self.hidden_layer_sizes,
                'learning_rate_init': self.learning_rate_init
            }
            max_attempts = self.max_attempts if self.early_stopping else self.max_iters
            self.fit_started_ = True
            fitted_weights, loss, _ = self.runner.run_one_experiment_(algorithm=self.algorithm,
                                                                      problem=problem,
                                                                      max_iters=self.max_iters,
                                                                      max_attempts=max_attempts,
                                                                      total_args=total_args,
                                                                      **params)

            # Save fitted weights
            self.fitted_weights = problem.get_state()
            self.loss = loss
            self.output_activation = self.fitness_fn.get_output_activation()

        return self

    def predict(self, x_test):
        if not np.shape(x_test)[1] == (self.node_list[0] - self.bias):
            raise Exception("""The number of columns in X must equal %d"""
                            % ((self.node_list[0] - self.bias),))

        y_pred, pp = self._predict(X=x_test,
                                   fitted_weights=self.fitted_weights,
                                   node_list=self.node_list,
                                   input_activation=self.activation,
                                   output_activation=self.output_activation,
                                   bias=self.bias)

        self.predicted_probabilities = pp
        return y_pred

    def predict_proba(self, x_test):
        self.predict(x_test)
        return self.predicted_probabilities
