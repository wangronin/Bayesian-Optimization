import os, logging

from . import AcquisitionFunction, Surrogate
from .BayesOpt import BO, ParallelBO, NoisyBO, AnnealingBO
from .Solution import Solution
from .Surrogate import RandomForest, GaussianProcess
from .SearchSpace import SearchSpace, OrdinalSpace, ContinuousSpace, NominalSpace
from .Extension import OptimizerPipeline

__all__ = [
    'BO', 'ParallelBO', 'NoisyBO', 'AnnealingBO', 'Solution',
    'AcquisitionFunction', 'Surrogate', 'RandomForest', 'GaussianProcess',
    'SearchSpace', 'OrdinalSpace', 'ContinuousSpace', 
    'NominalSpace', 'RandomForest', 'OptimizerPipeline'
]

# To use `dill` for the pickling, which works for
# much more python objects
os.environ['LOKY_PICKLER'] = 'dill' 

verbose = {
    False : logging.NOTSET,
    'DEBUG' : logging.DEBUG,
    'INFO' : logging.INFO
}

# TODO: add an interface function `fmin`
def fmin():
    pass
