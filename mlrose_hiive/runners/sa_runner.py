import mlrose_hiive
from mlrose_hiive.decorators import short_name
from mlrose_hiive.runners._runner_base import _RunnerBase

"""
Example usage:

    experiment_name = 'example_experiment'
    problem = TSPGenerator.generate(seed=SEED, number_of_cities=22)

    sa = SARunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=OUTPUT_DIRECTORY,
                  seed=SEED,
                  iteration_list=2 ** np.arange(14),
                  max_attempts=5000,
                  temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])

    # the two data frames will contain the results
    df_run_stats, df_run_curves = sa.run()                  
"""


@short_name('sa')
class SARunner(_RunnerBase):

    def __init__(self, problem, experiment_name, seed, iteration_list, temperature_list, decay_list=None,
                 max_attempts=500, generate_curves=True, **kwargs):
        super().__init__(problem=problem, experiment_name=experiment_name, seed=seed, iteration_list=iteration_list,
                         max_attempts=max_attempts, generate_curves=generate_curves,
                         **kwargs)
        self.temperature_list = temperature_list
        if decay_list is None:
            decay_list = [mlrose_hiive.GeomDecay]
        self.decay_list = decay_list

    def run(self):
        temperatures = [decay(init_temp=t) for t in self.temperature_list for decay in self.decay_list]
        return super().run_experiment_(algorithm=mlrose_hiive.simulated_annealing,
                                       schedule=('Temperature', temperatures))
