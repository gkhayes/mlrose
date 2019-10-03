""" MLROSe setup file."""

# Author: Genevieve Hayes
# Modified: Andrew Rollings
# License: BSD 3 clause

from setuptools import setup


def readme():
    """
    Function to read the long description for the MLROSe package.
    """
    with open('README.md') as _file:
        return _file.read()


setup(name='mlrose-hiive',
      version='1.2.0-ga',
      description="MLROSe: Machine Learning, Randomized Optimization and"
      + " Search",
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/hiive/mlrose',
      author='Genevieve Hayes (modifed by Andrew Rollings)',
      license='BSD',
      classifiers=[
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
          "Topic :: Software Development :: Libraries :: Python Modules"],
      packages=['mlrose','mlrose.runners','mlrose.generators', 'mlrose.algorithms',
                'mlrose.algorithms.decay', 'mlrose.algorithms.crossovers',
                'mlrose.opt_probs', 'mlrose.fitness', 'mlrose.algorithms.mutators',
                'mlrose.neural', 'mlrose.neural.activation'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas', 'networkx'],
      python_requires='>=3',
      zip_safe=False)
