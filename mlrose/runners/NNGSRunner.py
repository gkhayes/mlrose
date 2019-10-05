import mlrose.algorithms as mla
from mlrose.gridsearch.grid_search_mixin import GridSearchMixin

from mlrose.runners._RunnerBase import _RunnerBase
from mlrose.neural import NNClassifier


"""
Example usage:

    grid_search_parameters = ({
        'max_iters': [1, 2, 4, 8, 16, 32, 64, 128],                     # nn params
        'learning_rate': [0.001, 0.002, 0.003],                         # nn params
        'schedule': [ArithDecay(1), ArithDecay(100), ArithDecay(1000)]  # sa params
    })

    nnr = NNGSRunner(x_train=x_train,
                     y_train=y_train,
                     x_test=x_test,
                     y_test=y_test,
                     experiment_name='nn_test',
                     algorithm=mlrose.algorithms.sa.simulated_annealing,
                     algorithm_params=grid_search_parameters,
                     iteration_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                     hidden_nodes_set=[[4, 4, 4]],
                     activation_set=[mlrose.neural.activation.relu],
                     bias=True,
                     early_stopping=False,
                     clip_max=1e+10,
                     max_attempts=500,
                     generate_curves=True,
                     seed=200972)

    results = nnr.run()          # GridSearchCV instance returned    
"""


class NNGSRunner(_RunnerBase, GridSearchMixin):

    def __init__(self, x_train, y_train, x_test, y_test,
                 experiment_name, seed, iteration_list,
                 algorithm, algorithm_params,
                 hidden_nodes_set, activation_set, learning_rates=None, cv=5,
                 bias=True, early_stopping=False, clip_max=1e+10,
                 max_attempts=500, generate_curves=True):

        super().__init__(problem=None, experiment_name=experiment_name, seed=seed, iteration_list=iteration_list,
                         generate_curves=generate_curves)

        self.hidden_nodes_set = hidden_nodes_set
        self.activation_set = activation_set
        self.learning_rates = learning_rates
        self.bias = bias
        self.early_stopping = early_stopping
        self.clip_max = clip_max

        # algorithm grid-search params
        self.algorithm_params = algorithm_params

        # extract nn parameters
        self.parameters = {
            'hidden_nodes': self.hidden_nodes_set,
            'activation': self.activation_set,
            'learning_rate': self.learning_rates
        }
        self.parameters.update(algorithm_params)

        self.classifier = NNClassifier(runner=self,
                                       algorithm=algorithm,
                                       max_attempts=max_attempts)
        self.runner_name = f'nngs_{algorithm.__short_name__}'

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.cv = cv

    def run(self):
        sr = self._perform_grid_search(classifier=self.classifier,
                                       parameters=self.parameters,
                                       x_train=self.x_train,
                                       y_train=self.y_train,
                                       cv=self.cv)
        return sr

    def run_one_experiment_(self, algorithm, total_args=None, **params):
        if self._extra_args is not None and len(self._extra_args) > 0:
            params = {**params, **self._extra_args}
        user_info = [(k, v) for k, v in params.items() if k != 'problem']
        return self._invoke_algorithm(algorithm=algorithm,
                                      max_attempts=self.max_attempts,
                                      curve=self.generate_curves,
                                      user_info=user_info, **params)

