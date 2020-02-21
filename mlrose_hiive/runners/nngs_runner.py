import sklearn.metrics as skmt

from mlrose_hiive import NNClassifier
from mlrose_hiive.decorators import short_name
from mlrose_hiive.runners._nn_runner_base import _NNRunnerBase
from mlrose_hiive.decorators import get_short_name

"""
Example usage:
    from mlrose_hiive.runners import NNGSRunner

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
                     algorithm=mlrose_hiive.algorithms.sa.simulated_annealing,
                     grid_search_parameters=grid_search_parameters,
                     iteration_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                     hidden_layer_sizes=[[44,44]],
                     bias=True,
                     early_stopping=False,
                     clip_max=1e+10,
                     max_attempts=500,
                     generate_curves=True,
                     seed=200972)

    results = nnr.run()          # GridSearchCV instance returned    
"""


@short_name('nngs')
class NNGSRunner(_NNRunnerBase):

    def __init__(self, x_train, y_train, x_test, y_test, experiment_name, seed, iteration_list, algorithm,
                 grid_search_parameters, grid_search_scorer_method=skmt.balanced_accuracy_score,
                 bias=True, early_stopping=True, clip_max=1e+10,
                 max_attempts=500, n_jobs=1, cv=5, generate_curves=True, output_directory=None,
                 **kwargs):

        # update short name based on algorithm
        self._set_dynamic_runner_name(f'{get_short_name(self)}_{get_short_name(algorithm)}')

        # take a copy of the grid search parameters
        grid_search_parameters = {**grid_search_parameters}

        # hack for compatibility purposes
        if 'max_iter' in grid_search_parameters:
            grid_search_parameters['max_iter'] = grid_search_parameters.pop('max_iters')

        # call base class init
        super().__init__(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                         experiment_name=experiment_name,
                         seed=seed,
                         iteration_list=iteration_list,
                         grid_search_parameters=grid_search_parameters,
                         generate_curves=generate_curves,
                         output_directory=output_directory,
                         n_jobs=n_jobs,
                         cv=cv,
                         grid_search_scorer_method=grid_search_scorer_method,
                         **kwargs)

        # build the classifier
        self.classifier = NNClassifier(runner=self,
                                       algorithm=algorithm,
                                       max_attempts=max_attempts,
                                       clip_max=clip_max,
                                       early_stopping=early_stopping,
                                       seed=seed,
                                       bias=bias)

    def run_one_experiment_(self, algorithm, total_args, **params):
        if self._extra_args is not None and len(self._extra_args) > 0:
            params = {**params, **self._extra_args}

        total_args.update(params)
        total_args.pop('problem')
        user_info = [(k, v) for k, v in total_args.items()]

        return self._invoke_algorithm(algorithm=algorithm,
                                      curve=self.generate_curves,
                                      user_info=user_info,
                                      additional_algorithm_args=total_args,
                                      **params)
