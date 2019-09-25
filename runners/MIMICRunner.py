try:
    import mlrose
except:
    import sys
    sys.path.append("..")
    import mlrose
import time

from runners._RunnerBase import RunnerBase


class MIMICRunner(RunnerBase):

    def __init__(self, problem, seed, iteration_list, keep_percent_list, max_attempts=500, generate_curves=True):
        super().__init__(problem, seed, iteration_list, max_attempts, generate_curves)
        self.keep_percent_list = keep_percent_list

    def run(self):
        return super()._run_experiment(name='MIMIC',
                                       algorithm=mlrose.mimic,
                                       keep_pct=('Keep Percent', self.keep_percent_list))
