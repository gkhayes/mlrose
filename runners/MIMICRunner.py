try:
    import mlrose
except:
    import sys
    sys.path.append("..")
    import mlrose
import time

from runners._RunnerBase import _RunnerBase


class MIMICRunner(_RunnerBase):

    def __init__(self, problem, experiment_name, seed, iteration_list, keep_percent_list,
                 max_attempts=500, generate_curves=True, **kwargs):
        super().__init__(problem=problem, experiment_name=experiment_name, seed=seed, iteration_list=iteration_list,
                         max_attempts=max_attempts, generate_curves=generate_curves,
                         **kwargs)
        self.keep_percent_list = keep_percent_list

    def run(self):
        return super()._run_experiment(runner_name='MIMIC',
                                       algorithm=mlrose.mimic,
                                       keep_pct=('Keep Percent', self.keep_percent_list))
