# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:16:39 2018

@author: Hao Wang
@email: wangronin@gmail.com

"""
from pdb import set_trace

from typing import Callable, Any, Tuple
import os, sys, dill, functools, logging, time

import pandas as pd
import numpy as np
import json, copy, re 

from .base import baseBO
from .Solution import Solution
from .SearchSpace import SearchSpace
from . import InfillCriteria as IC
from .Surrogate import SurrogateAggregation
from .misc import proportional_selection, non_dominated_set_2d, bcolors, LoggerFormatter

class BO(baseBO):
    def pre_eval_check(self, X):
        """check for the duplicated solutions, as it is not allowed
        for noiseless objective functions
        """
        if not isinstance(X, Solution):
            X = Solution(X, var_name=self.var_names)
        
        N = X.N
        if hasattr(self, 'data'):
            X = X + self.data

        _ = []
        for i in range(N):
            x = X[i]
            idx = np.arange(len(X)) != i
            CON = np.all(np.isclose(np.asarray(X[idx][:, self.r_index], dtype='float'),
                                    np.asarray(x[self.r_index], dtype='float')), axis=1)
            INT = np.all(X[idx][:, self.i_index] == x[self.i_index], axis=1)
            CAT = np.all(X[idx][:, self.d_index] == x[self.d_index], axis=1)
            if not any(CON & INT & CAT):
                _ += [i]

        return X[_]
    
    # TODO: this might be generic enough to be part of the baseBO class
    def create_acquisition(self, plugin=None, dx=False):
        """
        plugin : float,
            the minimal objective value used in improvement-based infill criteria
            Note that it should be given in the original scale
        """
        plugin = 0 if plugin is None else (plugin - self.fmin) / self.frange
        kwargs = {
            'model' : self.model, 
            'plugin' : plugin, 
            'minimize' : self.minimize
        }
        kwargs.update(self.acquisition_par)

        if self.n_point > 1:       # multi-point acquisitions
            _t = np.exp(self.acquisition_par['t'] * np.random.randn())
            kwargs.update({'t' : _t})
            
        fun = getattr(IC, self._acquisition_fun)(**kwargs)
        fun = functools.partial(fun, dx=dx)
        return fun

class AnnealingBO(BO):
    def annealing(self):
        self.acquisition_par['t']
        pass

    def create_acquisition(self, plugin=None, dx=False):
        # TODO: to implement this
        """
        plugin : float,
            the minimal objective value used in improvement-based infill criteria
            Note that it should be given in the original scale
        """
        plugin = 0 if plugin is None else (plugin - self.fmin) / self.frange
        kwargs = {
            'model' : self.model, 
            'plugin' : plugin, 
            'minimize' : self.minimize
        }
        kwargs.update(self.acquisition_par)

        if self.n_point > 1:       # multi-point acquisitions
            _t = np.exp(self.acquisition_par['t'] * np.random.randn())
            kwargs.update({'t' : _t})
            
        fun = getattr(IC, self._acquisition_fun)(**kwargs)
        fun = functools.partial(fun, dx=dx)
        return fun

class SelfAdaptiveBO(BO):
    pass

class NoisyBO(BO):
    def create_acquisition(self, plugin=None, dx=False):
        # use the model prediction to determine the plugin under noisy scenarios
        if plugin is None:
            plugin = min(self.model.predict(self.data)) \
                if self.minimize else max(self.model.predict(self.data))
        
        return super().create_acquisition(plugin, dx)