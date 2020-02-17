import numpy as np

from mlrose_hiive import QueensOpt


class QueensGenerator:
    @staticmethod
    def generate(seed, size=20):
        np.random.seed(seed)
        problem = QueensOpt(length=size)
        return problem
