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

def obj_func(x):
    """Example mix-integer test function
    subject to minimization
    """
    x_r, x_i, x_d = np.array(x[:dim]), x[dim], x[dim + 1]
    if x_d == 'OK':
        tmp = 0
    else:
        tmp = 1
    return np.sum(x_r ** 2.) + abs(x_i - 10) / 123. + tmp * 2.

# Continuous variables can be specified as follows:
# a 5-D variable in [-5, 5]^5
C = ContinuousSpace([-5, 5]) * dim  

# Equivalently, you can also use
# C = ContinuousSpace([[-5, 5]]] * dim) 
# The general usage is:
# ContinuousSpace([[lb_1, ub_1], [lb_2, ub_2], ..., [lb_n, ub_n]]) 

# Integer (ordinal) variables can be specified as follows:
# The domain of integer variables can be given as with continuous ones
I = OrdinalSpace([-100, 100])

# Discrete (nominal) variables can be specified as follows:
# No lb, ub... a list of category instead
N = NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E'])

# The whole search space can be constructed:
search_space = C + I + N

# Bayesian optimization also uses a Surrogate model
# For mixed variable type, the random forest is typically used
model = RandomForest(levels=search_space.levels)

# The Bayesian optimization algorithm
# Please try to play with argument 'n_point' if your optimization 
# task is extremely slow (e.g., tuning machine learning algorithms)
opt = BO(search_space, obj_func, model, max_iter=n_step, 
         n_init_sample=n_init_sample, 
         n_point=1,       # number of the candidate solution proposed in each iteration
         n_job=1,         # number of processes for the parallel execution
         minimize=True, 
         verbose=True,    # turn this off, if you prefer no output
         optimizer='MIES')

opt.run()
