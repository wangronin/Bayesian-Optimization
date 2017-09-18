#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:57:47 2017

@author: wangronin
"""

import pdb

import numpy as np
from deap import benchmarks
from GaussianProcess_old import GaussianProcess_extra as GaussianProcess
from BayesOpt import BayesOpt, RandomForest, RrandomForest

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

thetaL = 1e-3 * (ub - lb) * np.ones(dim)
thetaU = 10 * (ub - lb) * np.ones(dim)
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

model = GaussianProcess(regr='constant', corr='matern',
                        theta0=theta0, thetaL=thetaL,
                        thetaU=thetaU, nugget=None,
                        nugget_estim=False, normalize=False,
                        verbose=False, random_start=15 * dim,
                        random_state=None, optimizer='BFGS')

# min_samples_leaf = max(1, int(n_init_sample / 20.))
# max_features = int(np.ceil(dim * 5 / 6.))
# model = RandomForest(n_estimators=100,
#                      max_features=max_features,
#                      min_samples_leaf=min_samples_leaf)

# model = RrandomForest()

search_space = [x1, x2]
opt = BayesOpt(search_space, obj_func, model, max_iter=n_step, random_seed=None,
               n_init_sample=n_init_sample, minimize=True, verbose=False, debug=True,
               optimizer='MIES')
               
opt.run()
