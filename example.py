#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:57:47 2017

@author: wangronin
"""

import pdb

import numpy as np
from deap import benchmarks
from GaussianProcess import GaussianProcess_extra as GaussianProcess
from BayesOpt import BayesOpt

np.random.seed(1)

dim = 2
n_step = 20
n_init_sample = 10
obj_func = lambda x: benchmarks.himmelblau(x)[0]
lb = np.array([-6] * dim)
ub = np.array([6] * dim)

x1 = {'name' : "x1",
      'type' : 'R',
      'bounds': [lb[0], ub[0]]}

x2 = {'name' : "x2",
      'type' : 'R',
      'bounds': [lb[1], ub[1]]}
    
search_space = [x1, x2]
opt = BayesOpt(search_space, obj_func, max_iter=n_step, random_seed=None,
               n_init_sample=n_init_sample, minimize=True, verbose=True)
               
opt.optimize()
