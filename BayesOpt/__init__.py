import os, logging

from . import InfillCriteria, Surrogate
from .BayesOpt import BO, ParallelBO, NoisyBO, AnnealingBO
from .Solution import Solution
from .Surrogate import RandomForest
from .SearchSpace import OrdinalSpace, ContinuousSpace, NominalSpace, from_dict
from .Extension import OptimizerPipeline

__all__ = [
    'BO', 'ParallelBO', 'NoisyBO', 'AnnealingBO', 'Solution', 'from_dict',
    'InfillCriteria', 'Surrogate', 'OrdinalSpace', 'ContinuousSpace', 
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
