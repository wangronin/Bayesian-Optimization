#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:57:47 2017

@author: wangronin
"""

import pdb

import numpy as np
from BayesOpt import BayesOpt
from BayesOpt.Surrogate import RandomForest
from BayesOpt.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

np.random.seed(123)

dim = 2
n_step = 10
n_init_sample = 15

def obj_func(x):
   x_r, x_i, x_d = np.array(x[:2]), x[2], x[3]
   if x_d == 'OK':
       tmp = 0
   else:
       tmp = 1
   return np.sum(x_r ** 2.) + abs(x_i - 10) / 123. + tmp * 2.

C = ContinuousSpace([-5, 5]) * 2
I = OrdinalSpace([-100, 100])
N = NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E'])

search_space = C * I * N

# thetaL = 1e-3 * (ub - lb) * np.ones(dim)
# thetaU = 10 * (ub - lb) * np.ones(dim)
# theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

# model = GaussianProcess(regr='constant', corr='matern',
#                         theta0=theta0, thetaL=thetaL,
#                         thetaU=thetaU, nugget=None,
#                         nugget_estim=False, normalize=False,
#                         verbose=False, random_start=15 * dim,
#                         random_state=None, optimizer='BFGS')

# min_samples_leaf = max(1, int(n_init_sample / 20.))
# max_features = int(np.ceil(dim * 5 / 6.))
# model = RandomForest(n_estimators=100,
#                      max_features=max_features,
#                      min_samples_leaf=min_samples_leaf)

model = RandomForest(levels=search_space.levels)
# model = RrandomForest(levels=search_space.levels, seed=1, max_features='sqrt')

opt = BayesOpt(search_space, obj_func, model, max_iter=n_step, random_seed=None,
               n_init_sample=n_init_sample, n_point=1, n_jobs=1, minimize=True, 
               verbose=True, debug=False, optimizer='MIES')

opt.run()

pdb.set_trace()
