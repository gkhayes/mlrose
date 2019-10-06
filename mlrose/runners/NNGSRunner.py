import time

from mlrose import short_name
from mlrose.gridsearch.grid_search_mixin import GridSearchMixin

from mlrose.runners._RunnerBase import _RunnerBase
from mlrose.neural import NNClassifier


"""
Example usage:
    from mlrose.runners import NNGSRunner
    
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
                     grid_search_parameters=grid_search_parameters,
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


@short_name('nngs')
class NNGSRunner(_RunnerBase, GridSearchMixin):

    def __init__(self, x_train, y_train, x_test, y_test,
                 experiment_name, seed, iteration_list,
                 algorithm, grid_search_parameters,
                 hidden_nodes_set, activation_set, learning_rates=None, cv=5,
                 bias=True, early_stopping=False, clip_max=1e+10,
                 max_attempts=500, generate_curves=True,
                 output_directory=None):

        super().__init__(problem=None, experiment_name=experiment_name, seed=seed, iteration_list=iteration_list,
                         generate_curves=generate_curves, output_directory=output_directory)

        self.hidden_nodes_set = hidden_nodes_set
        self.activation_set = activation_set
        self.learning_rates = learning_rates
        # self.bias = bias
        # self.early_stopping = early_stopping

        # algorithm grid-search params
        self.grid_search_parameters = grid_search_parameters

        # extract nn parameters
        self.parameters = {
            'hidden_nodes': self.hidden_nodes_set,
            'activation': self.activation_set,
            'learning_rate': self.learning_rates
        }
        self.parameters.update(grid_search_parameters)
        self.classifier = NNClassifier(runner=self,
                                       algorithm=algorithm,
                                       max_attempts=max_attempts,
                                       clip_max=clip_max,
                                       early_stopping=early_stopping,
                                       bias=bias)

        # update short name based on algorithm
        print(self.dynamic_runner_name())
        self._set_dynamic_runner_name(f'nngs_{algorithm.__short_name__}')
        print(self.dynamic_runner_name())
        print(self.runner_name())

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.cv = cv

    def run(self):
        self._setup()
        print(f'Running {self.dynamic_runner_name()}')

        run_start = time.perf_counter()
        sr = self._perform_grid_search(classifier=self.classifier,
                                       parameters=self.parameters,
                                       x_train=self.x_train,
                                       y_train=self.y_train,
                                       cv=self.cv)
        run_end = time.perf_counter()
        print(f'Run time: {run_end - run_start}')

        best_params = sr.best_params_
        best_score = sr.best_score_
        best_estimator = sr.best_estimator_
        best_loss = best_estimator.loss
        best_fitted_weights = best_estimator.fitted_weights  # ndarray
        edf = {
            'cv_results_df': sr.cv_results_
        }
        self._create_and_save_run_data_frames(extra_data_frames=edf)

        return sr

    def run_one_experiment_(self, algorithm, total_args, **params):
        if self._extra_args is not None and len(self._extra_args) > 0:
            params = {**params, **self._extra_args}
        if total_args is not None:
            total_args.update(params)

        user_info = [(k, v) for k, v in total_args.items() if k != 'problem']
        return self._invoke_algorithm(algorithm=algorithm,
                                      curve=self.generate_curves,
                                      user_info=user_info,
                                      additional_algorithm_args=total_args,
                                      **params)

