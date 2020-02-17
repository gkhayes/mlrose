import mlrose_hiive
from mlrose_hiive.decorators import short_name
from mlrose_hiive.runners._runner_base import _RunnerBase

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


@short_name('ga')
class GARunner(_RunnerBase):

    def __init__(self, problem, experiment_name, seed, iteration_list, population_sizes, mutation_rates,
                 hamming_factors=None, hamming_factor_decays=None, max_attempts=500, generate_curves=True, **kwargs):
        super().__init__(problem=problem, experiment_name=experiment_name, seed=seed, iteration_list=iteration_list,
                         max_attempts=max_attempts, generate_curves=generate_curves,
                         **kwargs)
        self.population_sizes = population_sizes
        self.mutation_rates = mutation_rates
        self.hamming_factors = hamming_factors
        self.hamming_factor_decays = hamming_factor_decays

    def run(self):
        return super().run_experiment_(algorithm=mlrose_hiive.genetic_alg,
                                       pop_size=('Population Size', self.population_sizes),
                                       mutation_prob=('Mutation Rate', self.mutation_rates),
                                       hamming_factor=('Hamming Factor', self.hamming_factors),
                                       hamming_decay_factor=('Hamming Factor Decay Rate', self.hamming_factor_decays))




