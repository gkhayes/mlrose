
try:
    import mlrose
except:
    import sys
    sys.path.append("..")
    import mlrose
from mlrose.runners._RunnerBase import _RunnerBase

"""
Example usage:

    experiment_name = 'example_experiment'
    problem = TSPGenerator.generate(seed=SEED, number_of_cities=22)

    ga = GARunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=SEED,
                  iteration_list=2 ** np.arange(12),
                  max_attempts=1000,
                  population_sizes=[150, 200, 300],
                  mutation_rates=[0.4, 0.5, 0.6])
                  
    # the two data frames will contain the results
    df_run_stats, df_run_curves = ga.run()                  
"""


class GARunner(_RunnerBase):

    def __init__(self, problem, experiment_name, seed, iteration_list, population_sizes, mutation_rates,
                 max_attempts=500, generate_curves=True, **kwargs):
        super().__init__(problem=problem, experiment_name=experiment_name, seed=seed, iteration_list=iteration_list,
                         max_attempts=max_attempts, generate_curves=generate_curves,
                         **kwargs)
        self.population_sizes = population_sizes
        self.mutation_rates = mutation_rates

    def run(self):
        return super()._run_experiment(runner_name='GA',
                                       algorithm=mlrose.genetic_alg,
                                       pop_size=('Population Size', self.population_sizes),
                                       mutation_prob=('Mutation Rate', self.mutation_rates))




