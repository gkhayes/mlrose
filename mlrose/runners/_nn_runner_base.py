import time
from abc import ABC

import pandas as pd
import numpy as np
from joblib.my_exceptions import WorkerInterrupt

from mlrose import GridSearchMixin
from mlrose.decorators import get_short_name
from mlrose.runners._runner_base import _RunnerBase


class _NNRunnerBase(_RunnerBase, GridSearchMixin, ABC):

    _interrupted_result_list = []

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
        self.cv_results_df = None

    def run(self):
        try:
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

            # pull the stats from the best estimator to here.
            # (as grid search will have cloned this object).
            self.__dict__.update(sr.best_estimator_.runner.__dict__)

            # dump the results to disk
            self.cv_results_df = self._make_cv_results_data_frame(sr.cv_results_)
            edf = {
                'cv_results_df': self.cv_results_df
            }
            self._create_and_save_run_data_frames(extra_data_frames=edf)

            try:
                self._dump_pickle_to_disk(sr, 'grid_search_results')
            except:
                pass

            try:
                y_pred = sr.best_estimator_.predict(self.x_test)
                score = self.score(y_pred=y_pred, y_true=self.y_train)
                self._print_banner(f'Score: {score}')
            except:
                pass
            return self.run_stats_df, self.curves_df, self.cv_results_df, sr
        except WorkerInterrupt:
            return None, None, None, None
        finally:
            self._tear_down()

        """
        best = {
            'best_params': sr.best_params_,
            'best_score': sr.best_score_,
            'best_estimator': sr.best_estimator_,
            'best_loss': sr.best_estimator_.best_loss_,
            'best_fitted_weights': sr.best_estimator_.fitted_weights  # ndarray
        }
        """

    @staticmethod
    def _make_cv_results_data_frame(cv_results):
        cv_results = cv_results.copy()
        param_prefix = 'param_'
        # drop params
        param_labels = [k for k in cv_results if param_prefix in k]
        # clean_results = {k: v for k, v in cv_results.items() if 'param_' not in k}

        new_param_values = {p: [] for p in param_labels}
        for v in cv_results['params']:
            for p in param_labels:
                pl = p.replace(param_prefix, '')
                new_param_values[p].append(_NNRunnerBase._sanitize_value(v[pl]))

        # replace values with sanitized values
        cv_results.update(new_param_values)
        df = pd.DataFrame(cv_results)
        df.dropna(inplace=True)
        return df

    @staticmethod
    def _sanitize_value(value):
        return get_short_name(value) if not isinstance(value, tuple) and not isinstance(value, list) else str(value)

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

    def _grid_search_score_intercept(self, y_true, y_pred, sample_weight=None, adjusted=False):
        if not self.classifier.fit_started_ and self.has_aborted():
            return np.NaN
        return super()._grid_search_score_intercept(y_true=y_true,
                                                    y_pred=y_pred,
                                                    sample_weight=sample_weight,
                                                    adjusted=adjusted)
