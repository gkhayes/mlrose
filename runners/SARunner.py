try:
    import mlrose
except:
    import sys
    sys.path.append("..")
    import mlrose
import time

from runners._RunnerBase import RunnerBase


class SARunner(RunnerBase):

    def __init__(self, problem, seed, iteration_list, temperature_list, max_attempts=500, generate_curves=True):
        super().__init__(problem, seed, iteration_list, max_attempts, generate_curves)
        self.temperature_list = temperature_list

    def run(self):
        temperatures = [mlrose.GeomDecay(init_temp=t) for t in self.temperature_list]
        return super()._run_experiment(name='SA',
                                       algorithm=mlrose.simulated_annealing,
                                       schedule=('Temperature', temperatures))
