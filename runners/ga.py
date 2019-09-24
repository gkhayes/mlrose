from runners.base import RunnerBase

try:
    import mlrose
except:
    import sys
    sys.path.append("..")
    import mlrose
import time

import numpy as np


class GARunner(RunnerBase):

    def __init__(self, problem, seed,
                 iteration_list, population_sizes, mutation_rates,
                 max_attempts=500, generate_curves=True):
        self.iterations = iteration_list
        self.population_sizes = population_sizes
        self.mutation_rates = mutation_rates
        self.problem = problem
        self.seed = seed
        self.max_attempts = max_attempts
        self.generate_curves = generate_curves
        self._setup()

    def _setup(self):
        self.raw_run_stats = []
        self.best_states = []
        self.fitness_curves = []
        self.run_stats_df = None

    def iteration_callback_(self, iteration, done, state, fitness, curve, user_data):
        if iteration in self.iterations or done:
            end = time.perf_counter()
            population, mutation_rate, start = user_data
            t = end - start
            print(f'iteration:[{iteration}], done:[{done}], time:[{t:.2f}]')
            print(f'[population:[{population}], mutation_rate:[{mutation_rate:.2f}], fitness:[{fitness:.8f}]')
            print(f'\t{state}')
            print()

            remaining_iterations = list(filter(lambda x: x >= iteration, self.iterations))
            iterations = [min(remaining_iterations)] if not done else remaining_iterations
            for i in iterations:
                run_stat = {
                    'Mutation Probability': mutation_rate,
                    'Best Fitness': fitness,
                    'Iterations': i,
                    'Time': t,
                    'Population Size': population,
                }

                self.raw_run_stats.append(run_stat)
                if curve is not None:
                    self.fitness_curves.append(list(curve))
                self.best_states.append(state)
        return True

    def run(self):
        i = int(max(self.iterations))
        self._setup()

        for pc in self.population_sizes:
            for mr in self.mutation_rates:
                np.random.seed(self.seed)
                start = time.perf_counter()
                mlrose.genetic_alg(self.problem,
                                   mutation_prob=mr,
                                   max_attempts=self.max_attempts,
                                   pop_size=pc,
                                   max_iters=i,
                                   curve=self.generate_curves,
                                   random_state=self.seed,
                                   state_fitness_callback=self.iteration_callback_,
                                   callback_user_info=(pc, mr, start))
