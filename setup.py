""" MLROSe setup file.

    Author: Genevieve Hayes
    License: 3-clause BSD license.
"""

from setuptools import setup

setup(name = 'mlrose',
      version = '0.1.0',
      description = "MLROSe: Machine Learning, Randomized Optimization and"
                    + " Search",
      long_description = "MLROSe provides classes and functions for solving"
                    + " a range of well known AI/ML optimization problems," 
                    + " including N-Queens, Travelling Salesperson and finding"
                    + " the optimal weights for a neural network. The package"
                    + " includes implementations of the hill climbing,"
                    + " randomized hill climbing, simulated annealing,"
                    + " genetic algorithm and MIMIC algorithms, and includes"
                    + " functionality for both pre-defined and user-defined"
                    + " fitness functions.",
      url = 'https://github.com/gkhayes/mlrose',
      author = 'Genevieve Hayes',
      license = 'BSD',
      classifiers = [
              "Intended Audience :: Education",
              "Intended Audience :: Science/Research",
              "Natural Language :: English",
              "License :: OSI Approved :: BSD License",
              "Programming Language :: Python",
              "Programming Language :: Python :: 3",
              "Topic :: Scientific/Engineering",
              "Topic :: Scientific/Engineering :: Artificial Intelligence",
              "Topic :: Scientific/Engineering :: Mathematics",
              "Topic :: Software Development :: Libraries",
              "Topic :: Software Development :: Libraries :: Python Modules",
                     ],
      packages = ['mlrose'],
      install_requires = ['numpy', 'scipy', 'sklearn'],
      python_requires = '>=3',
      zip_safe = False)
