from abc import ABC, abstractmethod
import time
import itertools as it
import numpy as np
import pandas as pd


class RunnerBase(ABC):

    def __init__(self, problem, seed, iteration_list, max_attempts=500, generate_curves=True):
        self.problem = problem
        self.seed = seed
        self.iteration_list = iteration_list
        self.max_attempts = max_attempts
        self.generate_curves = generate_curves
        self.run_stats_df = None
        self.curves_df = None
        self._initial_fitness = None
        self.parameter_description_dict = {}

    def _setup(self):
        self._raw_run_stats = []
        self._fitness_curves = []
        pass

    def _run_experiment(self, name, algorithm, **kwargs):
        self._setup()

        # extract loop params
        values = [([(k, v) for v in vs]) for (k, (n, vs)) in kwargs.items()]
        self.parameter_description_dict = {k: n for (k, (n, _)) in kwargs.items()}
        value_sets = list(it.product(*values))
        run_start = time.perf_counter()
        i = int(max(self.iteration_list))

        print(f'Running {name}')
        for vns in value_sets:
            self.current_args = dict(vns)
            np.random.seed(self.seed)
            self.iteration_start_time = time.perf_counter()
            algorithm(problem=self.problem,
                      max_attempts=self.max_attempts,
                      curve=self.generate_curves,
                      random_state=self.seed,
                      max_iters=i,
                      state_fitness_callback=self._save_state,
                      callback_user_info=None,
                      **self.current_args)
        run_end = time.perf_counter()
        print(f'Run time: {run_end - run_start}')

        self.run_stats_df = pd.DataFrame(self._raw_run_stats)
        self.curves_df = pd.DataFrame(self._fitness_curves)

        return self.run_stats_df, self.curves_df

    @staticmethod
    def _create_curve_stat(iteration, fitness, param_stats):
        curve_stat = {
            'Iteration': iteration,
            'Fitness': fitness
         }
        curve_stat.update(param_stats)
        return curve_stat

    def _save_state(self, iteration, done, state, fitness, curve, user_data):
        if iteration == 1:
            self._initial_fitness = fitness  # 1.0 / fitness

        if iteration not in self.iteration_list and not done:
            return True

        end = time.perf_counter()

        t = end - self.iteration_start_time
        print(f'iteration:[{iteration}], done:[{done}], time:[{t:.2f}]')
        if user_data is not None and len(user_data) > 0:
            data_desc = ', '.join([f'{n}: [{v}] ' for (n, v) in user_data])
            print(data_desc)
        print(f'\t{state}')
        print()

        remaining_iterations = list(filter(lambda x: x >= iteration, self.iteration_list))
        iterations = [min(remaining_iterations)] if not done else remaining_iterations
        gd = lambda n: n if n not in self.parameter_description_dict.keys() else self.parameter_description_dict[n]

        param_stats = {gd(k): self.current_args[k] for k in self.current_args}
        for i in iterations:
            run_stat = {
                'Iterations': i,
                'Best Fitness': fitness,  # 1.0 / fitness,
                'Time': t,
                'Best State': state
            }
            run_stat.update(param_stats)

            self._raw_run_stats.append(run_stat)

        if curve is not None and (done or iteration == max(self.iteration_list)):
            fc = list([(0, self._initial_fitness)]) + list(zip(range(1, iteration + 1),
                                                               [f for f in curve]))  # [1.0 / f for f in curve]))

            curve_stats = [self._create_curve_stat(i, v, param_stats) for (i, v) in fc]
            self._fitness_curves.extend(curve_stats)
        return True

    @abstractmethod
    def run(self):
        pass
