""" Classes for defining algorithms problem objects."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

from .ga import (genetic_alg)
from .sa import (simulated_annealing)
from .hc import (hill_climb)
from .rhc import (random_hill_climb)
from .gd import (gradient_descent)
from .mimic import (mimic)

from .crossovers import UniformCrossOver, TSPCrossOver, OnePointCrossOver

from .decay import ArithDecay, CustomSchedule, ExpDecay, GeomDecay
from .mutators import ChangeOneMutator, DiscreteMutator, ShiftOneMutator, SwapMutator
