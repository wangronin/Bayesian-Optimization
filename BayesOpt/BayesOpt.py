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
from joblib import Parallel, delayed

from . import InfillCriteria as IC
from .base import baseBO
from .Solution import Solution
from .SearchSpace import SearchSpace
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

# TODO: add other Parallelization options: 1) niching-based 2) Pareto-front of PI-EI
class ParallelBO(BO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.n_point > 1:
            if 't' not in self._acquisition_par:
                self._acquisition_par['t'] = getattr(IC, self._acquisition_fun)().t
            # TODO: add support for UCB as well

    def _batch_arg_max_acquisition(self, n_point, plugin, return_dx):
        criteria = []
        for _ in range(n_point):
            _t = np.exp(self._acquisition_par['t'] * np.random.randn())
            _acquisition_par = copy.copy(self._acquisition_par).update({'t' : _t})
            criteria.append(
                self._create_acquisition(plugin, return_dx, _acquisition_par)
            )
        
        if self.n_job > 1:
            __ = Parallel(n_jobs=self.n_job)(
                delayed(self._argmax_restart)(c) for c in criteria
            )
        else:
            __ = [list(self._argmax_restart(_)) for _ in criteria]
        
        return tuple(zip(*__))

class AnnealingBO(ParallelBO):
    def __init__(self, t0=2, tf=1e-1, schedule='exp', *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        self.t0 = t0
        self.tf = tf
        self.schedule = schedule
        self._acquisition_par['t'] = t0
        
        max_iter = (self.max_FEs - self._DoE_size) / self.n_point
        if self.schedule == 'exp':                          # exponential
            self.alpha = (self.tf / t0) ** (1. / max_iter) 
            self._anealer = lambda t: t * self.alpha
        elif self.schedule == 'linear':
            self.eta = (t0 - self.tf) / max_iter            # linear
            self._anealer = lambda t: t - self.eta
        elif self.schedule == 'log':
            self.c = self.tf * np.log(max_iter + 1)         # logarithmic 
            self._anealer = lambda t: t * self.c / np.log(self.iter_count + 2)
        else:
            raise NotImplementedError
        
        def callback():
            self._acquisition_par['t'] = self._anealer(self._acquisition_par['t'])
        self._acquisition_callbacks += [callback]
    
class SelfAdaptiveBO(ParallelBO):
    def __init__(self, *argv, **kwargs):
        super.__init__(*argv, **kwargs)
        assert self.n_point > 1

    def _batch_arg_max_acquisition(self, n_point, plugin, return_dx):
        criteria = []
        _t_list = []
        N = int(n_point / 2)

        for _ in range(n_point):
            _t = np.exp(self._acquisition_par['t'] * np.random.randn())
            _t_list.append(_t)
            _acquisition_par = copy.copy(self._acquisition_par).update({'t' : _t})
            criteria.append(
                self._create_acquisition(plugin, return_dx, _acquisition_par)
            )
        
        if self.n_job > 1:
            __ = Parallel(n_jobs=self.n_job)(
                delayed(self._argmax_restart)(c) for c in criteria
            )
        else:
            __ = [list(self._argmax_restart(_)) for _ in criteria]
        
        # NOTE: this adaptation is different from what I did in the LeGO paper..
        idx = np.argsort(__[1])[::-1][:N]
        self._acquisition_par['t'] = np.mean([_t_list[i] for i in idx])
        return tuple(zip(*__))

class NoisyBO(ParallelBO):
    # TODO: implement the strategy for re-evaluation
    def pre_eval_check(self, X):
        pass

    def _create_acquisition(self, plugin=None, return_dx=False, acquisition_par=None):
        # use the model prediction to determine the plugin under noisy scenarios
        if plugin is None:
            plugin = min(self.model.predict(self.data)) \
                if self.minimize else max(self.model.predict(self.data))
        
        return super()._create_acquisition(plugin, return_dx, acquisition_par)
