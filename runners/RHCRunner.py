try:
    import mlrose
except:
    import sys
    sys.path.append("..")
    import mlrose
import time

import numpy as np
import pandas as pd

from runners._RunnerBase import _RunnerBase


class RHCRunner(_RunnerBase):

    def __init__(self, problem, experiment_name, seed, iteration_list, restart_list,
                 max_attempts=500, generate_curves=True, **kwargs):
        super().__init__(problem=problem, experiment_name=experiment_name, seed=seed, iteration_list=iteration_list,
                         max_attempts=max_attempts, generate_curves=generate_curves,
                         **kwargs)
        self.restart_list = restart_list

    def run(self):
        return super()._run_experiment(runner_name='RHC',
                                       algorithm=mlrose.random_hill_climb,
                                       restarts=('Restarts', self.restart_list))
