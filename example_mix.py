#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:57:47 2017

@author: wangronin
"""

import pdb

import numpy as np
from GaussianProcess_old import GaussianProcess_extra as GaussianProcess
from BayesOpt import BayesOpt, RandomForest, RrandomForest

np.random.seed(1)

dim = 2
n_step = 20
n_init_sample = 15

def obj_func(x):
   x_r, x_i, x_d = np.array(x[:2]), x[2], x[3]
   if x_d == 'OK':
       tmp = 0
   else:
       tmp = 1
   return np.sum(x_r ** 2.) + abs(x_i - 10) / 123. + tmp * 2.

x1 = {'name' : "x1",
      'type' : 'R',
      'bounds': [-5, 5]}
x2 = {'name' : "x2",
      'type' : 'R',
      'bounds': [-5, 5]}
x3 = {'name' : "x3",
      'type' : 'I',
      'bounds': [-100, 100]}
x4 = {'name' : "x4",
      'type' : 'D',
      'levels': ['OK', 'A', 'B', 'C', 'D', 'E']}

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

model = RrandomForest()

search_space = [x1, x2, x3, x4]
opt = BayesOpt(search_space, obj_func, model, max_iter=n_step, random_seed=None,
               n_init_sample=n_init_sample, minimize=True, verbose=True, debug=False,
               optimizer='MIES')
               
opt.run()
