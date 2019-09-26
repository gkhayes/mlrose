try:
    import mlrose
except:
    import sys
    sys.path.append("..")
    import mlrose
import time

from runners._RunnerBase import _RunnerBase


class SARunner(_RunnerBase):

    def __init__(self, problem, experiment_name, seed, iteration_list, temperature_list,
                 max_attempts=500, generate_curves=True, **kwargs):
        super().__init__(problem=problem, experiment_name=experiment_name, seed=seed, iteration_list=iteration_list,
                         max_attempts=max_attempts, generate_curves=generate_curves,
                         **kwargs)
        self.temperature_list = temperature_list

    def run(self):
        temperatures = [mlrose.GeomDecay(init_temp=t) for t in self.temperature_list]
        return super()._run_experiment(runner_name='SA',
                                       algorithm=mlrose.simulated_annealing,
                                       schedule=('Temperature', temperatures))
