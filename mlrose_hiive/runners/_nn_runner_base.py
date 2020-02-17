import time
import hashlib
import os
from abc import ABC

import pandas as pd
import numpy as np
import pickle as pk

from joblib.my_exceptions import WorkerInterrupt

from mlrose_hiive import GridSearchMixin
from mlrose_hiive.runners._runner_base import _RunnerBase


class _NNRunnerBase(_RunnerBase, GridSearchMixin, ABC):
    _interrupted_result_list = []

    def __init__(self, x_train, y_train, x_test, y_test,
                 experiment_name, seed, iteration_list,
                 grid_search_parameters,
                 grid_search_scorer_method,
                 cv=5,
                 generate_curves=True,
                 output_directory=None,
                 verbose_grid_search=True,
                 n_jobs=1,
                 replay=False,
                 **kwargs):
        # call super on _RunnerBase
        _RunnerBase.__init__(self, problem=None, experiment_name=experiment_name, seed=seed,
                             iteration_list=iteration_list,
                             generate_curves=generate_curves, output_directory=output_directory,
                             replay=replay,
                             copy_zero_curve_fitness_from_first=True)

        # call super on GridSearchMixin
        GridSearchMixin.__init__(self, scorer_method=grid_search_scorer_method)

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
        self.best_params = None

    def run(self):
        try:
            self._setup()
            print(f'Running {self.dynamic_runner_name()}')
            if self.replay_mode():
                gsr_name = f"{super()._get_pickle_filename_root('grid_search_results')}.p"
                with open(gsr_name, 'rb') as pickle_file:
                    sr = pk.load(pickle_file)
            else:
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

            self.best_params = sr.best_params_
            # dump the results to disk
            self.cv_results_df = self._make_cv_results_data_frame(sr.cv_results_)
            edf = {
                'cv_results_df': self.cv_results_df
            }
            self._create_and_save_run_data_frames(extra_data_frames=edf, final_save=True)

            try:
                self._dump_pickle_to_disk(sr, 'grid_search_results', final_save=True)
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

    def _get_pickle_filename_root(self, name):
        filename_root = super()._get_pickle_filename_root(name)
        arg_text = ''.join([f'{k}_{self._sanitize_value(v)}_'
                            for k, v in self._current_logged_algorithm_args.items()
                            if 'state' not in k])
        arg_hash = f'__{hashlib.md5(arg_text.encode()).hexdigest()}'.upper() if len(arg_text) > 0 else ''
        filename_root += arg_hash
        return filename_root

    def _tear_down(self):
        if self.best_params is None or self.replay_mode() is None:
            super()._tear_down()
            return
        filename_root = super()._get_pickle_filename_root('')

        path = os.path.join(*filename_root.split('/')[:-1])
        filename_part = filename_root.split('/')[-1]
        if path[0] != '/':
            path = f'{path}'
        # find all data frames output by this runner
        filenames = [fn for fn in os.listdir(path) if (filename_part in fn
                                                       and fn.endswith('.p')
                                                       and '_df_' in fn)]
        # get the best parameters
        df_best_params = pd.DataFrame([{k: self._sanitize_value(v) for k, v in self.best_params.items()}])

        # file the files that match the best parameters (and don't)
        correct_files = []
        incorrect_files = []
        for fn in filenames:
            filename = os.path.join(path, fn)
            with open(filename, 'rb') as pickle_file:
                try:
                    df = pk.load(pickle_file)
                    delete = (pd.merge(df, df_best_params, how='inner')).empty
                    if delete:
                        incorrect_files.append(filename)
                    else:
                        correct_files.append(filename)
                except:
                    pass

        # extract the md5s from the names for the best and non-best parameter files
        correct_md5s = list(set([p.split('_')[-1][:-2] for p in correct_files]))
        incorrect_md5s = list(set([p.split('_')[-1][:-2] for p in incorrect_files]))

        # remove the suboptimal files
        all_incorrect_files = []
        for incorrect_md5 in incorrect_md5s:
            all_incorrect_files.extend([os.path.join(path, fn) for fn in os.listdir(path) if incorrect_md5 in fn])

        for filename in all_incorrect_files:
            os.rename(filename, f'{filename}.del')
            # os.remove(filename)

        # rename the best files by removing the md5 from the end
        all_correct_files = []
        for correct_md5 in correct_md5s:
            all_correct_files.extend([(os.path.join(path, fn), f'__{correct_md5}')
                                      for fn in os.listdir(path)
                                      if correct_md5 in fn])

        for filename, correct_md5 in all_correct_files:
            correct_filename = filename.replace(correct_md5, '')
            if os.path.exists(correct_filename):
                os.rename(correct_filename, f'{correct_filename}.bak')
            os.rename(filename, correct_filename)

        super()._tear_down()

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
