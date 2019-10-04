""" MLROSe initialization file."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

from .algorithms import (hill_climb, random_hill_climb, simulated_annealing, genetic_alg, mimic)
from .algorithms.decay import GeomDecay, ArithDecay, ExpDecay, CustomSchedule
from .algorithms.crossovers import OnePointCrossOver, UniformCrossOver, TSPCrossOver
from .algorithms.mutators import ChangeOneMutator, DiscreteMutator, SwapMutator, ShiftOneMutator
from .fitness import (OneMax, FlipFlop, FourPeaks, SixPeaks, ContinuousPeaks,
                      Knapsack, TravellingSales, Queens, MaxKColor, 
                      CustomFitness)
from .neural import NeuralNetwork, LinearRegression, LogisticRegression, CoreNeuralNetwork
from .neural.activation import (identity, relu, sigmoid, softmax, tanh)
from .neural.fitness import NetworkWeights
from .neural.utils import (gradient_descent)
from .neural.utils.weights import (flatten_weights, unflatten_weights)

from .opt_probs import DiscreteOpt, ContinuousOpt, KnapsackOpt, TSPOpt, QueensOpt, FlipFlopOpt, MaxKColorOpt

from .runners import GARunner, MIMICRunner, RHCRunner, SARunner
from .runners import (build_data_filename)
from .generators import KnapsackGenerator, TSPGenerator, FlipFlopGenerator, QueensGenerator, MaxKColorGenerator
