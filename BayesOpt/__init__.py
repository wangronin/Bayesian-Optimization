# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:45:21 2015

@author: wangronin
"""
import os, logging

from . import InfillCriteria, Surrogate
from .BayesOpt import BO, NoisyBO, AnnealingBO
from .Solution import Solution
from .Surrogate import RandomForest
from .SearchSpace import OrdinalSpace, ContinuousSpace, NominalSpace, from_dict

__all__ = ['BO', 'NoisyBO', 'AnnealingBO', 'Solution', 'from_dict',
           'InfillCriteria', 'Surrogate', 'OrdinalSpace', 'ContinuousSpace', 
           'NominalSpace', 'RandomForest']

# To use `dill` for the pickling, which works for
# much more python objects
os.environ['LOKY_PICKLER'] = 'dill' 

verbose = {
    False : logging.NOTSET,
    'DEBUG' : logging.DEBUG,
    'INFO' : logging.INFO
}