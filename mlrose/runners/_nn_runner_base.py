import time
from abc import ABC, abstractmethod

import pandas as pd

from mlrose import GridSearchMixin
from mlrose.runners._runner_base import _RunnerBase


class _NNRunnerBase(_RunnerBase, GridSearchMixin, ABC):

    def __init__(self, x_train, y_train, x_test, y_test,
                 experiment_name, seed, iteration_list,
                 grid_search_parameters,
                 cv=5,
                 generate_curves=True,
                 output_directory=None,
                 verbose_grid_search=True,
                 n_jobs=1,
                 **kwargs):
        super().__init__(problem=None, experiment_name=experiment_name, seed=seed, iteration_list=iteration_list,
                         generate_curves=generate_curves, output_directory=output_directory,
                         copy_zero_curve_fitness_from_first=True)

        self.classifier = None

        # add algorithm grid-search params
        self.grid_search_parameters = self.build_grid_search_parameters(grid_search_parameters=grid_search_parameters,
                                                                        **kwargs)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose_grid_search = verbose_grid_search

    def temp(self):
        pass

    def run(self):
        self._setup()
        print(f'Running {self.dynamic_runner_name()}')

        run_start = time.perf_counter()
        sr = self._perform_grid_search(classifier=self.classifier,
                                       parameters=self.grid_search_parameters,
                                       x_train=self.x_train,
                                       y_train=self.y_train,
                                       cv=self.cv,
                                       n_jobs=self.n_jobs,
                                       verbose=self.verbose_grid_search)
        run_end = time.perf_counter()
        print(f'Run time: {run_end - run_start}')

        # dump the results to disk
        self._dump_pickle_to_disk(sr, 'grid_search_results')
        edf = {
            'cv_results_df': pd.DataFrame(sr.cv_results_)
        }
        self._create_and_save_run_data_frames(extra_data_frames=edf)

        # pull the stats from the best estimator to here.
        # (as grid search will have cloned this object).
        self.__dict__.update(sr.best_estimator_.runner.__dict__)

        try:
            y_pred = sr.best_estimator_.predict(self.x_test)
            score = self.score(y_pred=y_pred, y_true=self.y_train)
            self._print_banner(f'Score: {score}')
        except:
            pass

        """
        best = {
            'best_params': sr.best_params_,
            'best_score': sr.best_score_,
            'best_estimator': sr.best_estimator_,
            'best_loss': sr.best_estimator_.best_loss_,
            'best_fitted_weights': sr.best_estimator_.fitted_weights  # ndarray
        }
        """
        return sr

    @staticmethod
    def build_grid_search_parameters(grid_search_parameters, **kwargs):
        # extract nn parameters
        all_grid_search_parameters = {
            # 'hidden_nodes': hidden_nodes_set,
            # 'activation': activation_set,
            # 'learning_rate': learning_rates
        }
        all_grid_search_parameters.update(grid_search_parameters)
        all_grid_search_parameters.update(**kwargs)
        return all_grid_search_parameters
