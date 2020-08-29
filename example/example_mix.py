#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:57:47 2017

@author: wangronin
@email: wangronin@gmail.com
        h.wang@liacs.leidenuniv.nl
"""

from pdb import set_trace

import numpy as np
from BayesOpt import BO
from BayesOpt.Surrogate import RandomForest
from BayesOpt.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

np.random.seed(666)

dim = 2
n_step = 50
n_init_sample = 10 * dim
lb, ub = -5, 5
eval_type = 'dict' # control the type of parameters for evaluation: dict | list

def obj_func_list_eval(x):
    """Example mix-integer test function
    subject to minimization
    with a list as input
    """
    x_r, x_i, x_d = np.array(x[:dim]), x[dim], x[dim + 1]
    if x_d == 'OK':
        tmp = 0
    else:
        tmp = 1
    return np.sum(x_r ** 2.) + abs(x_i - 10) / 123. + tmp * 2.

def obj_func_dict_eval(par):
    """Example mix-integer test function
    with dictionary as input 
    """
    x_r = np.asarray([par[k] for k in par.keys() if k.startswith('continuous')])
    x_i, x_d = par['ordinal'], par['nominal']
    if x_d == 'OK':
        tmp = 0
    else:
        tmp = 1
    return np.sum(x_r ** 2.) + abs(x_i - 10) / 123. + tmp * 2.

obj_func = obj_func_list_eval if eval_type == 'list' else obj_func_dict_eval

# Continuous variables can be specified as follows:
# a 2-D variable in [-5, 5]^2
# for 2 variables, the naming scheme is continuous0, continuous1
C = ContinuousSpace([lb, ub], var_name='continuous') * dim  

# Equivalently, you can also use
# C = ContinuousSpace([[-5, 5]]] * dim) 
# The general usage is:
# ContinuousSpace([[lb_1, ub_1], [lb_2, ub_2], ..., [lb_n, ub_n]]) 

# Integer (ordinal) variables can be specified as follows:
# The domain of integer variables can be given as with continuous ones
# var_name is optional
I = OrdinalSpace([-100, 100], var_name='ordinal')

# Discrete (nominal) variables can be specified as follows:
# No lb, ub... a list of categories instead
N = NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E'], var_name='nominal')

# The whole search space can be constructed:
search_space = C + I + N

# Bayesian optimization also uses a Surrogate model
# For mixed variable type, the random forest is typically used
model = RandomForest(levels=search_space.levels)

# For continuous variable type only, Gaussian process regression (GPR) is suggested
# # this is a standard setting. no need to change
# mean = constant_trend(dim, beta=0)    

# # autocorrelation parameters of GPR
# thetaL = 1e-10 * (ub - lb) * np.ones(dim)
# thetaU = 2 * (ub - lb) * np.ones(dim)
# theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL
# model = GaussianProcess(mean=mean, corr='matern',
#                         theta0=theta0, thetaL=thetaL, thetaU=thetaU,
#                         nugget=1e-10, noise_estim=False,
#                         optimizer='BFGS', wait_iter=5, random_start=15 * dim,
#                         likelihood='concentrated', eval_budget=150 * dim)

# The Bayesian optimization algorithm
# Please try to play with argument 'n_point' if your optimization 
# task is extremely slow (e.g., tuning machine learning algorithms)
opt = BO(search_space, obj_func, model, max_iter=n_step, 
         n_init_sample=n_init_sample, 
         n_point=3,        # number of the candidate solution proposed in each iteration
         n_job=3,          # number of processes for the parallel execution
         minimize=True, 
         eval_type=eval_type, # use this parameter to control the type of evaluation
         verbose=True,     # turn this off, if you prefer no output
         optimizer='MIES')

xopt, fitness, stop_dict = opt.run()

print('xopt: {}'.format(xopt))
print('fopt: {}'.format(fitness))
print('stop criteria: {}'.format(stop_dict))