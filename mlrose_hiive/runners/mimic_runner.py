import mlrose_hiive
from mlrose_hiive.decorators import short_name
from mlrose_hiive.runners._runner_base import _RunnerBase

"""
Example usage:

    experiment_name = 'example_experiment'
    problem = TSPGenerator.generate(seed=SEED, number_of_cities=22)

    mmc = MIMICRunner(problem=problem,
                      experiment_name=experiment_name,
                      output_directory=OUTPUT_DIRECTORY,
                      seed=SEED,
                      iteration_list=2 ** np.arange(10),
                      max_attempts=500,
                      keep_percent_list=[0.25, 0.5, 0.75])
                      
    # the two data frames will contain the results
    df_run_stats, df_run_curves = mmc.run()
"""


@short_name('mimic')
class MIMICRunner(_RunnerBase):

    def __init__(self, problem, experiment_name, seed, iteration_list, population_sizes,
                 keep_percent_list, max_attempts=500, generate_curves=True, use_fast_mimic=False, **kwargs):
        super().__init__(problem=problem, experiment_name=experiment_name, seed=seed, iteration_list=iteration_list,
                         max_attempts=max_attempts, generate_curves=generate_curves,
                         **kwargs)
        self.keep_percent_list = keep_percent_list
        self.population_sizes = population_sizes
        self._use_fast_mimic = None
        if hasattr(problem, 'set_mimic_fast_mode') and callable(getattr(problem, 'set_mimic_fast_mode')):
            self._use_fast_mimic = use_fast_mimic
            problem.set_mimic_fast_mode(use_fast_mimic)

    def _setup(self):
        super()._setup()
        if self._use_fast_mimic is not None:
            self._log_current_argument('use_fast_mimic', self._use_fast_mimic)

    def run(self):
        return super().run_experiment_(algorithm=mlrose_hiive.mimic,
                                       pop_size=('Population Size', self.population_sizes),
                                       keep_pct=('Keep Percent', self.keep_percent_list))
