#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:57:47 2017

@author: wangronin
"""
import pdb

import os
import pandas as pd
from mpi4py import MPI
import numpy as np

from deap import benchmarks
from BayesOpt import BayesOpt
from BayesOpt.surrogate import RrandomForest, RandomForest
from BayesOpt.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
runs = comm.Get_size()

def obj_func(x):
   x_r, x_i, x_d = np.array(x[:2]), x[2], x[3]
   if x_d == 'OK':
       tmp = 0
   else:
       tmp = 1
   return np.sum(x_r ** 2.) + abs(x_i - 10) / 123. + tmp * 2.

def create_optimizer(dim, fitness, n_step, n_init_sample, model_type):

    C = ContinuousSpace([-5, 5]) * 2
    I = OrdinalSpace([-100, 100])
    N = NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E'])

    search_space = C * I * N
    levels = search_space.levels
    
    if model_type == 'GP':
#        thetaL = 1e-3 * (ub - lb) * np.ones(dim)
#        thetaU = 10 * (ub - lb) * np.ones(dim)
#        theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL
#    
#        model = GaussianProcess(regr='constant', corr='matern',
#                                theta0=theta0, thetaL=thetaL,
#                                thetaU=thetaU, nugget=1e-5,
#                                nugget_estim=False, normalize=False,
#                                verbose=False, random_start = 15*dim,
#                                random_state=None)
        pass
                               
    elif model_type == 'sklearn-RF':
        min_samples_leaf = max(1, int(n_init_sample / 20.))
        max_features = int(np.ceil(dim * 5 / 6.))
        model = RandomForest(levels=levels, n_estimators=100,
                            max_features=max_features,
                            min_samples_leaf=min_samples_leaf)

    elif model_type == 'R-RF':
        min_samples_leaf = max(1, int(n_init_sample / 20.))
        max_features = int(np.ceil(dim * 5 / 6.))
        model = RrandomForest(levels=levels, n_estimators=100, 
                              max_features=max_features,
                              min_samples_leaf=min_samples_leaf)

    opt = BayesOpt(search_space, fitness, model, max_iter=n_step, random_seed=None,
                   n_init_sample=n_init_sample, minimize=True, optimizer='MIES')
    
    return opt

dim = 4
n_step = 20
n_init_sample = 30
model_type = 'R-RF'
functions = {"mine": obj_func}

# generate, distribute and set the random seeds for reproducibility
if rank == 0:
    np.random.seed(1)
    seed = np.random.randint(0, 65535, runs)

    if not os.path.exists('./data'):
        os.makedirs('./data')
else:
    seed = None
seed = comm.scatter(seed, root=0)
np.random.seed(seed)

    
for func_name, func in functions.items():    
    if rank == 0:
        print("testing on function: {} dim: {}".format(func_name, dim))
        
    y_hist_best = np.zeros((n_step, runs))
    
    csv_name = './data/{}D-{}N-{}.csv'.format(dim, n_init_sample, func_name)
    opt = create_optimizer(dim, func, n_step, n_init_sample, model_type)
    opt.run()
    hist_perf = opt.hist_perf

    comm.Barrier()
    __ = comm.gather(hist_perf, root=0)

    if rank == 0:
        data = np.atleast_2d(__)
        data = data.T if data.shape[1] != runs else data
        mean_ = np.mean(data, axis=1)
        error_ = np.std(data, axis=1, ddof=1) / np.sqrt(runs)
        print('mean : %f'%mean_)
        print('std error: %f'%error_)
        
        # append the new data the csv
        df = pd.DataFrame(data)
        df = pd.DataFrame(data, columns=['run{}'.format(_+1) for _ in range(runs)])
        df.to_csv(csv_name, mode='w', header=True, index=False)