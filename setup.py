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
      version='2.1.0',
      description="MLROSe: Machine Learning, Randomized Optimization and"
      + " Search",
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/hiive/mlrose',
      author='Genevieve Hayes (modified by Andrew Rollings)',
      license='BSD',
      download_url='https://github.com/hiive/mlrose/archive/2.1.0.tar.gz',
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
                'mlrose.neural', 'mlrose.neural.activation', 'mlrose.neural.fitness',
                'mlrose.neural.utils', 'mlrose.decorators',
                'mlrose.gridsearch'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas', 'networkx', 'joblib'],
      python_requires='>=3',
      zip_safe=False)
