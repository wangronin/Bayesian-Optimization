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
from BayesOpt import configurator

np.random.seed(1)

dim = 2
n_step = 20
n_init_sample = 10
obj_func = lambda x: benchmarks.himmelblau(x)[0]
lb = np.array([-6] * dim)
ub = np.array([6] * dim)

#thetaL = 1e-3 * (ub - lb) * np.ones(dim)
#thetaU = 10 * (ub - lb) * np.ones(dim)
#theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL
#
#model = GaussianProcess(regr='constant', corr='matern',
#                        theta0=theta0, thetaL=thetaL,
#                        thetaU=thetaU, nugget=1e-5,
#                        nugget_estim=False, normalize=False,
#                        verbose=False, random_start=15 * dim,
#                        random_state=None)


x1 = {'name' : "x1",
      'type' : 'R',
      'bounds': [lb[0], ub[0]]}

x2 = {'name' : "x2",
      'type' : 'R',
      'bounds': [lb[1], ub[1]]}
    
search_space = [x1, x2]
opt = configurator(search_space, obj_func, 29, random_seed=None,
                    n_init_sample=n_init_sample, minimize=True,
                    verbose=True)

opt.optimize()
# for n in range(n_step):
#     xopt, fopt = opt.step()
    
#     print 'iteration {}:'.format(n + 1)
#     print 'xopt: {}'.format(xopt)
#     print 'fopt: {}'.format(fopt)
#     print
