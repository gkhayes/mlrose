try:
    import mlrose
except:
    import sys
    sys.path.append("..")
    import mlrose
import time

import numpy as np


class GARunner:

    def __init__(self, problem, seed, iteration_list, population_sizes, mutation_rates,
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

    def iteration_callback_(self, iteration, state, fitness, curve, user_data):
        if iteration in self.iterations:
            end = time.perf_counter()
            population, mutation_rate, start = user_data
            t = end - start
            print(f'[population:{population}, mutation_rate:{mutation_rate:.2f}, '
                  f'iteration:{iteration}] - fitness:{fitness:.4f}, time:{t:.2f}')
            print(f'\t{state}')
            print()

            run_stat = {
                'Mutation Probability': mutation_rate,
                'Best Fitness': fitness,
                'Iterations': iteration,
                'Time': t,
                'Population Size': population,
            }

            self.raw_run_stats.append(run_stat)
            if curve is not None:
                self.fitness_curves.append(list(curve))
            self.best_states.append(state)
        return True
        pass

    def run_ga(self):
        iters = max(self.iterations)
        self._setup()

        for pc in self.population_sizes:
            for mr in self.mutation_rates:
                np.random.seed(self.seed)
                start = time.perf_counter()
                mlrose.genetic_alg(self.problem,
                                   mutation_prob=mr,
                                   max_attempts=self.max_attempts,
                                   pop_size=pc,
                                   max_iters=int(iters),
                                   curve=self.generate_curves,
                                   random_state=self.seed,
                                   state_fitness_callback=self.iteration_callback_,
                                   callback_user_info=(pc, mr, start))
