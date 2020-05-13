#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:57:47 2017

@author: wangronin
"""

import pdb

import numpy as np
from deap import benchmarks

from GaussianProcess import GaussianProcess
from GaussianProcess.trend import constant_trend

from BayesOpt import BO
from BayesOpt.Surrogate import RandomForest
from BayesOpt.SearchSpace import ContinuousSpace
from BayesOpt.base import Solution

np.random.seed(42)

dim = 5
n_step = 50
n_init_sample = 21
obj_func = lambda x: benchmarks.ackley(x)[0]
lb = np.array([-6] * dim)
ub = np.array([6] * dim)

search_space = ContinuousSpace(list(zip(lb, ub)))

# Bayesian optimization also uses a Surrogate model
# For the continuous variable, the Gaussian process regression (GPR) is typically used

# trend function of GPR
# this is a standard setting. no need to change
mean = constant_trend(dim, beta=0)    

# autocorrelation parameters of GPR
thetaL = 1e-10 * (ub - lb) * np.ones(dim)
thetaU = 2 * (ub - lb) * np.ones(dim)
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

model = GaussianProcess(mean=mean, corr='matern',
                        theta0=theta0, thetaL=thetaL, thetaU=thetaU,
                        nugget=1e-10, noise_estim=False,
                        optimizer='BFGS', wait_iter=5, random_start=30 * dim,
                        likelihood='concentrated', eval_budget=200 * dim)

# warm_data = Solution([2.7839273017368615, 1.1839018103644392], var_name=["r_0", "r_1"], n_eval=1, fitness=12.18884801)
# warm_data += Solution([-4.127776314690761, -4.128065755965569], var_name=["r_0", "r_1"], n_eval=1, fitness=38.61525121)


opt = BO(search_space, obj_func, model, max_iter=n_step,
         n_init_sample=n_init_sample, minimize=True, verbose=True, 
         wait_iter=10, 
         n_point=3,
         n_job=3,
         #warm_data="example-wd.csv",
        #  warm_data=warm_data,
         #warm_data=None,
         optimizer='BFGS'  # when using GPR model, 'BFGS' is faster than 'MIES'        
         )
               
opt.run()

# opt._initialize() 
# while not opt.check_stop():
#     opt.step()

#     eval_count = opt.eval_count

#     # you can put fopt in an array
#     fopt = opt.fopt

