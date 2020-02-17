from abc import ABC, abstractmethod
import time
import os
import itertools as it
import numpy as np
import pandas as pd
import pickle as pk
import inspect as lk
import signal
import multiprocessing
import ctypes

from mlrose_hiive.decorators import get_short_name
from mlrose_hiive.runners.utils import build_data_filename


class _RunnerBase(ABC):
    __abort = multiprocessing.Value(ctypes.c_bool)
    __spawn_count = multiprocessing.Value(ctypes.c_uint)
    __replay = multiprocessing.Value(ctypes.c_bool)
    __original_sigint_handler = None
    __sigint_params = None

    @classmethod
    def runner_name(cls):
        return get_short_name(cls)

    def dynamic_runner_name(self):
        return self.__dynamic_short_name__ if hasattr(self, '__dynamic_short_name__') else self.runner_name()

    def _set_dynamic_runner_name(self, name):
        self.__dynamic_short_name__ = name

    @staticmethod
    def _print_banner(text):
        print('*' * len(text))
        print(text)
        print('*' * len(text))

    @staticmethod
    def _sanitize_value(value):
        if isinstance(value, tuple) or isinstance(value, list):
            v = str(value)
        elif isinstance(value, np.ndarray):
            v = str(list(value))
        else:
            v = get_short_name(value)
        return v

    @abstractmethod
    def run(self):
        pass

    def __init__(self, problem, experiment_name, seed, iteration_list, max_attempts=500,
                 generate_curves=True, output_directory=None, copy_zero_curve_fitness_from_first=False, replay=False,
                 **kwargs):
        self.problem = problem
        self.seed = seed
        self.iteration_list = iteration_list
        self.max_attempts = max_attempts
        self.generate_curves = generate_curves
        self.parameter_description_dict = {}

        self.run_stats_df = None
        self.curves_df = None
        self._raw_run_stats = []
        self._fitness_curves = []
        self._curve_base = 0
        self._copy_zero_curve_fitness_from_first = copy_zero_curve_fitness_from_first
        self._copy_zero_curve_fitness_from_first_original = copy_zero_curve_fitness_from_first
        self._extra_args = kwargs
        self._output_directory = output_directory
        self._experiment_name = experiment_name
        self._current_logged_algorithm_args = {}
        self._run_start_time = None
        self._iteration_times = []
        if replay:
            self.set_replay_mode()
        self._increment_spawn_count()

    def _increment_spawn_count(self):
        with self.__spawn_count.get_lock():
            self.__spawn_count.value += 1

    def _decrement_spawn_count(self):
        with self.__spawn_count.get_lock():
            self.__spawn_count.value -= 1

    def _get_spawn_count(self):
        self._print_banner(f'*** Spawn Count Remaining: {self.__spawn_count.value} ***')
        return self.__spawn_count.value

    def abort(self):
        self._print_banner('*** ABORTING ***')
        with self.__abort.get_lock():
            self.__abort.value = True

    def has_aborted(self):
        return self.__abort.value

    def set_replay_mode(self, value=True):
        with self.__replay.get_lock():
            self.__replay.value = value

    def replay_mode(self):
        return self.__replay.value

    def _setup(self):
        self._raw_run_stats = []
        self._fitness_curves = []
        self._curve_base = 0

        self._iteration_times = []
        self._copy_zero_curve_fitness_from_first = self._copy_zero_curve_fitness_from_first_original
        self._current_logged_algorithm_args.clear()
        if self._output_directory is not None:
            if not os.path.exists(self._output_directory):
                os.makedirs(self._output_directory)

        # set up ctrl-c handler
        if self.__original_sigint_handler is None:
            self.__original_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._ctrl_c_handler)

    def _ctrl_c_handler(self, sig, frame):
        print('Interrupted - saving progress so far')
        self.__sigint_params = (sig, frame)
        self.abort()

    def _tear_down(self):
        # restore ctrl-c handler
        self._decrement_spawn_count()
        if self.__original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self.__original_sigint_handler)
            if self.has_aborted() and self._get_spawn_count() == 0:
                sig, frame = self.__sigint_params
                self.__original_sigint_handler(sig, frame)

    def _log_current_argument(self, arg_name, arg_value):
        self._current_logged_algorithm_args[arg_name] = arg_value

    def run_experiment_(self, algorithm, **kwargs):
        self._setup()
        # extract loop params
        values = [([(k, v) for v in vs]) for (k, (n, vs)) in kwargs.items() if vs is not None]
        self.parameter_description_dict = {k: n for (k, (n, vs)) in kwargs.items() if vs is not None}
        value_sets = list(it.product(*values))

        print(f'Running {self.dynamic_runner_name()}')
        run_start = time.perf_counter()
        for vns in value_sets:
            total_args = dict(vns)
            if 'max_iters' not in total_args:
                total_args['max_iters'] = int(max(self.iteration_list))

            self._run_one_experiment(algorithm, total_args)

        run_end = time.perf_counter()
        print(f'Run time: {run_end - run_start}')

        self._create_and_save_run_data_frames(final_save=True)
        self._tear_down()

        return self.run_stats_df, self.curves_df

    def _run_one_experiment(self, algorithm, total_args, **params):
        if self._extra_args is not None and len(self._extra_args) > 0:
            total_args.update(self._extra_args)
        total_args.update(params)

        user_info = [(k, v) for k, v in total_args.items()]
        self._invoke_algorithm(algorithm=algorithm, problem=self.problem,
                               max_attempts=self.max_attempts, curve=self.generate_curves,
                               user_info=user_info, **total_args)

    def _create_and_save_run_data_frames(self, extra_data_frames=None, final_save=False):
        self.run_stats_df = pd.DataFrame(self._raw_run_stats)
        self.curves_df = pd.DataFrame(self._fitness_curves)
        if self._output_directory is not None:
            if len(self.run_stats_df) > 0:
                self._dump_df_to_disk(self.run_stats_df, df_name='run_stats_df', final_save=final_save)
            if self.generate_curves and len(self.curves_df) > 0:
                self._dump_df_to_disk(self.curves_df, df_name='curves_df', final_save=final_save)
            # output any extra
            if isinstance(extra_data_frames, dict):
                for n, v in extra_data_frames.items():
                    self._dump_df_to_disk(v, df_name=n, final_save=final_save)

    def _dump_df_to_disk(self, df, df_name, final_save=False):
        filename_root = self._dump_pickle_to_disk(object_to_pickle=df,
                                                  name=df_name)
        df.to_csv(f'{filename_root}.csv')
        if final_save:
            print(f'Saving: [{filename_root}.csv]')

    def _get_pickle_filename_root(self, name):
        filename_root = build_data_filename(output_directory=self._output_directory,
                                            runner_name=self.dynamic_runner_name(),
                                            experiment_name=self._experiment_name,
                                            df_name=name)
        return filename_root

    def _dump_pickle_to_disk(self, object_to_pickle, name, final_save=False):
        if self._output_directory is None:
            return
        filename_root = self._get_pickle_filename_root(name)

        pk.dump(object_to_pickle, open(f'{filename_root}.p', "wb"))
        if final_save:
            print(f'Saving: [{filename_root}.p]')
        return filename_root

    def _load_pickles(self):
        curves_df_filename = f"{self._get_pickle_filename_root('curves_df')}.p"
        run_stats_df_filename = f"{self._get_pickle_filename_root('run_stats_df')}.p"
        self.curves_df = None
        self.run_stats_df = None
        if os.path.exists(curves_df_filename):
            with open(curves_df_filename, 'rb') as pickle_file:
                try:
                    self.curves_df = pk.load(pickle_file)
                except:
                    pass
        if os.path.exists(run_stats_df_filename):
            with open(run_stats_df_filename, 'rb') as pickle_file:
                try:
                    self.run_stats_df = pk.load(pickle_file)
                except:
                    pass

        return self.curves_df is not None and self.run_stats_df is not None

    def _invoke_algorithm(self, algorithm, problem, max_attempts,
                          curve, user_info, additional_algorithm_args=None, **total_args):
        self._current_logged_algorithm_args.update(total_args)
        if additional_algorithm_args is not None:
            self._current_logged_algorithm_args.update(additional_algorithm_args)

        if self.replay_mode() and self._load_pickles():
            return None, None, None

        # arg_text = [get_short_name(v) for v in self._current_logged_algorithm_args.values()]
        self._print_banner('*** Run START ***')
        np.random.seed(self.seed)

        valid_args = [k for k in lk.signature(algorithm).parameters]
        args_to_pass = {k: v for k, v in total_args.items() if k in valid_args}

        self._start_run_timing()
        ret = algorithm(problem=problem,
                        max_attempts=max_attempts,
                        curve=curve,
                        random_state=self.seed,
                        state_fitness_callback=self._save_state,
                        callback_user_info=user_info,
                        **args_to_pass)
        self._print_banner('*** Run END ***')
        self._curve_base = len(self._fitness_curves)
        return ret

    def _start_run_timing(self):
        self._run_start_time = time.perf_counter()

    @staticmethod
    def _create_curve_stat(iteration, curve_value, curve_data, t=None):
        curve_stat = {
            'Iteration': iteration,
            'Time': t,
            'Fitness': curve_value
        }

        curve_stat.update(curve_data)
        if isinstance(curve_value, dict):
            curve_stat.update(curve_value)
        return curve_stat

    def _save_state(self, iteration, state, fitness, user_data, attempt=0, done=False, curve=None):

        # log iteration timing
        end = time.perf_counter()
        t = end - self._run_start_time
        self._iteration_times.append(t)

        # do we need to log anything else?
        if iteration > 0 and iteration not in self.iteration_list and not done:
            return True

        display_data = {**self._current_logged_algorithm_args}
        if user_data is not None and len(user_data) > 0:
            display_data.update({n: v for (n, v) in user_data})
            data_desc = ', '.join([f'{n}:[{get_short_name(v)}]' for n, v in display_data.items()])
            print(data_desc)
        print(f'runner_name:[{self.dynamic_runner_name()}], experiment_name:[{self._experiment_name}], ' +
              ('' if attempt is None else f'attempt:[{attempt}], ') +
              f'iteration:[{iteration}], done:[{done}], '
              f'time:[{t:.2f}], fitness:[{fitness:.4f}]')

        state_string = str(state).replace('\n', '//')[:200]
        print(f'\t{state_string}...')
        print()

        gd = lambda n: n if n not in self.parameter_description_dict.keys() else self.parameter_description_dict[n]

        current_iteration_stats = {str(gd(k)): self._sanitize_value(v)
                                   for k, v in self._current_logged_algorithm_args.items()}
        current_iteration_stats.update({str(gd(k)): self._sanitize_value(v)
                                        for k, v in {k: v for (k, v) in user_data}.items()})

        # check for additional info
        gi = lambda k, v: {} if not hasattr(v, 'get_info__') else v.get_info__(t)
        ai = (gi(k, v) for k, v in current_iteration_stats.items())
        additional_info = {k: self._sanitize_value(v) for d in ai for k, v in d.items()}

        if iteration > 0:
            remaining_iterations = [i for i in self.iteration_list if i >= iteration]
            iterations = [min(remaining_iterations)] if not done else remaining_iterations
        else:
            iterations = [0]

        for i in iterations:
            run_stat = {
                'Iteration': i,
                'Fitness': fitness,
                'Time': t,
                'State': self._sanitize_value(state)
            }
            run_stat.update(additional_info)
            run_stat.update(current_iteration_stats)
            self._raw_run_stats.append(run_stat)

        if self.generate_curves and curve is not None:  # and (done or iteration == max(self.iteration_list)):
            curve_stats_saved = len(self._fitness_curves)
            total_curve_stats = self._curve_base + len(curve)
            curve_stats_to_save = total_curve_stats - curve_stats_saved

            fc = list(zip(range(curve_stats_saved, total_curve_stats + 1), curve[-curve_stats_to_save:]))

            curve_stats = [self._create_curve_stat(iteration=i,
                                                   curve_value=f,
                                                   curve_data=current_iteration_stats,
                                                   t=self._iteration_times[i]) for (i, f) in fc]

            self._fitness_curves.extend(curve_stats)

            if self._copy_zero_curve_fitness_from_first and len(self._fitness_curves) > 1:
                self._fitness_curves[0]['Fitness'] = self._fitness_curves[1]['Fitness']
                self._copy_zero_curve_fitness_from_first = False
            self._create_and_save_run_data_frames()

        # save progress
        # if iteration > 0:
        #    self._create_and_save_run_data_frames()

        return not (self.has_aborted() or done)
