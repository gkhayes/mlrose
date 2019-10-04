from abc import ABC, abstractmethod
import time
import os
import itertools as it
import numpy as np
import pandas as pd
import pickle as pk

from mlrose.runners.utils import build_data_filename


class _RunnerBase(ABC):

    @classmethod
    @abstractmethod
    def runner_name(cls):
        pass

    @abstractmethod
    def run(self):
        pass

    def __init__(self, problem, experiment_name, seed, iteration_list, max_attempts=500,
                 generate_curves=True, output_directory=None, **kwargs):
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
        self._zero_curve_stat = None
        self._extra_args = kwargs
        self._output_directory = output_directory
        self._experiment_name = experiment_name

    def _setup(self):
        self._raw_run_stats = []
        self._fitness_curves = []
        if self._output_directory is not None:
            if not os.path.exists(self._output_directory):
                os.makedirs(self._output_directory)
        """
        directory = "./data/old/curves/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        path1 = './data/old'
        path2 = './data/old/curves'
        """
        pass

    def run_experiment_(self, algorithm, save_data=True, **kwargs):
        self._setup()

        # extract loop params
        values = [([(k, v) for v in vs]) for (k, (n, vs)) in kwargs.items() if vs is not None]
        self.parameter_description_dict = {k: n for (k, (n, vs)) in kwargs.items() if vs is not None}
        value_sets = list(it.product(*values))
        run_start = time.perf_counter()
        i = int(max(self.iteration_list))

        print(f'Running {self.runner_name()}')
        for vns in value_sets:
            self.current_args = dict(vns)
            total_args = self.current_args.copy()

            if self._extra_args is not None and len(self._extra_args) > 0:
                total_args.update(self._extra_args)

            user_info = [(n, total_args[n]) for n in total_args]

            np.random.seed(self.seed)
            self.iteration_start_time = time.perf_counter()
            print(f'*** Iteration START - params: {total_args}')
            self._invoke_algorithm(algorithm, i, total_args, user_info)
            print(f'*** Iteration END - params: {total_args}')
            print()

        run_end = time.perf_counter()
        print(f'Run time: {run_end - run_start}')

        self.run_stats_df = pd.DataFrame(self._raw_run_stats)
        self.curves_df = pd.DataFrame(self._fitness_curves)

        if save_data and self._output_directory is not None:
            self._dump_df_to_disk(self.run_stats_df,
                                  df_name='run_stats_df')
            if self.generate_curves:
                self._dump_df_to_disk(self.curves_df,
                                      df_name='curves_df')

        return self.run_stats_df, self.curves_df

    def _invoke_algorithm(self, algorithm, i, total_args, user_info):
        algorithm(problem=self.problem,
                  max_attempts=self.max_attempts,
                  curve=self.generate_curves,
                  random_state=self.seed,
                  max_iters=i,
                  state_fitness_callback=self._save_state,
                  callback_user_info=user_info,
                  **total_args)

    def _dump_df_to_disk(self, df, df_name):
        filename_root = build_data_filename(output_directory=self._output_directory,
                                            runner_name=self.runner_name(),
                                            experiment_name=self._experiment_name,
                                            df_name=df_name)

        pk.dump(df, open(f'{filename_root}.p', "wb"))
        df.to_csv(f'{filename_root}.csv')

    @staticmethod
    def _create_curve_stat(iteration, fitness, curve_data, t=None):
        curve_stat = {
            'Iteration': iteration,
            'Time': t,
            'Fitness': fitness
        }
        curve_stat.update(curve_data)
        return curve_stat

    def _save_state(self, iteration, state, fitness, user_data, attempt=0, done=False, curve=None):

        if iteration > 0 and iteration not in self.iteration_list and not done:
            return True

        end = time.perf_counter()
        t = end - self.iteration_start_time

        if user_data is not None and len(user_data) > 0:
            data_desc = ', '.join([f'{n}:[{v}]' for (n, v) in user_data])
            print(data_desc)
        print(f'runner_name:[{self.runner_name()}], experiment_name:[{self._experiment_name}], ' +
              ('' if attempt is None else f'attempt:[{attempt}], ') +
              f'iteration:[{iteration}], done:[{done}], '
              f'time:[{t:.2f}], fitness:[{fitness:.4f}]')
        print(f'\t{state}')
        print()

        eval = lambda n: n if not hasattr(n, 'evaluate') else n.evaluate(t)
        gd = lambda n: n if n not in self.parameter_description_dict.keys() else self.parameter_description_dict[n]

        param_stats = {str(gd(k)): self.current_args[k] for k in self.current_args}
        # check for additional info
        gi = lambda k, v: {} if not hasattr(v, 'get_info__') else v.get_info__(k)
        ai = (gi(k, self.current_args[k]) for k in self.current_args)
        additional_info = {k: v for d in ai for k, v in d.items()}


        # additional_info.update(kv) for kv in [gi(k, self.current_args[k]) for k in self.current_args]

        all_stats = {**{p: eval(v) for (p, v) in user_data if p.lower() not in [k.lower() for k in param_stats.keys()]},
                     **param_stats, **additional_info}
        all_stats.update(additional_info)

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
                'State': state
            }
            run_stat = {**run_stat, **all_stats}

            self._raw_run_stats.append(run_stat)

        if self.generate_curves and iteration == 0:
            curve_stat = self._create_curve_stat(0, fitness, all_stats, t)
            self._zero_curve_stat = curve_stat

        if self.generate_curves and curve is not None and (done or iteration == max(self.iteration_list)):
            fc = list(zip(range(1, iteration + 1), curve))

            curve_stats = [self._zero_curve_stat] + [self._create_curve_stat(i, f, all_stats) for (i, f) in fc]

            # interpolate the time over the curve where we don't have times
            # use the timings from the current run stats (not strictly 100% accurate, but close enough)
            timings = [(r['Iteration'], r['Time']) for r in self._raw_run_stats
                       if all([r[x] == all_stats[x] for x in all_stats])]

            for t in range(1, len(timings)):
                i1, t1 = timings[t - 1]
                i2, t2 = timings[t]
                t_range = np.linspace(t1, t2, 1 + (i2 - i1))
                for i in range(i1, min(1 + i2, len(curve_stats))):
                    curve_stats[i]['Time'] = t_range[i - i1]
            self._fitness_curves.extend(curve_stats)

        return not done
