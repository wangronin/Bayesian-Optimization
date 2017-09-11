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
from GaussianProcess import GaussianProcess_extra as GaussianProcess
from BayesOpt import BayesOpt

np.random.seed(1)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
runs = comm.Get_size()

def create_optimizer(dim, fitness, lb, ub, n_step, n_init_sample):
    
    seed = int(os.getpid())
    x1 = {'name' : "x1",
          'type' : 'R',
          'bounds': [-6, 6]}

    x2 = {'name' : "x2",
          'type' : 'R',
          'bounds': [-6, 6]}
    
    search_space = [x1, x2]
    opt = BayesOpt(search_space, fitness, n_step + n_init_sample, random_seed=seed,
                        n_init_sample=n_init_sample, minimize=True)
    
    return opt


dims = [2]
n_step = 2
n_init_sample = 10
benchmarkfunctions = {
                #"schwefel":benchmarks.schwefel,
                #"ackley":benchmarks.himmelblau,
                #"rastrigin":benchmarks.rastrigin,
                #"bohachevsky":benchmarks.bohachevsky,
                #"schaffer":benchmarks.schaffer,
                "himmelblau": benchmarks.himmelblau
                }

for dim in dims:
    lb = np.array([-6] * dim)
    ub = np.array([6] * dim)
    
    for func_name, func in benchmarkfunctions.iteritems():    
        if rank == 0:
            print "testing on function:", func_name, "dim:", dim
            
        fitness = lambda x: func(x)[0]
        y_hist_best = np.zeros((n_step, runs))
        
        csv_name = './data/{}D-{}N-{}.csv'.format(dim, n_init_sample, func_name)
        opt = create_optimizer(dim, fitness, lb, ub, n_step, n_init_sample)
        opt.optimize()
        hist_perf = opt.hist_perf

        comm.Barrier()
        __ = comm.gather(fopt, root=0)

        if rank == 0:
            data = np.atleast_2d(__)
            print data
            data = data.T if data.shape[1] != runs else data
            mean_ = np.mean(data, axis=0)
            error_ = np.std(data, axis=0, ddof=1) / np.sqrt(runs)
            print 'mean : ', mean_
            print 'std error: ', error_
            
            # append the new data the csv
            df = pd.DataFrame(data)
            df = pd.DataFrame(data, columns=['run{}'.format(_+1) for _ in range(runs)])
            df.to_csv(csv_name, mode='w', header=True, index=False)