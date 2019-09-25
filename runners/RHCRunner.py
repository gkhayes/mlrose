try:
    import mlrose
except:
    import sys
    sys.path.append("..")
    import mlrose
import time

import numpy as np
import pandas as pd

from runners._RunnerBase import RunnerBase


class RHCRunner(RunnerBase):

    def __init__(self, problem, seed, iteration_list, restart_list, max_attempts=500, generate_curves=True):
        super().__init__(problem, seed, iteration_list, max_attempts, generate_curves)
        self.restart_list = restart_list

    def run(self):
        return super()._run_experiment(name='RHC',
                                       algorithm=mlrose.random_hill_climb,
                                       restarts=('Restarts', self.restart_list))
