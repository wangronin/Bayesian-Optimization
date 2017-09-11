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
from ego import ego
from GaussianProcess import GaussianProcess_extra as GaussianProcess
from Configurator import 

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
    conf = configurator(search_space, fitness, n_step, 
                        n_init_sample=n_init_sample, minimize=True)
    
    return opt


dims = [2]
n_step = 20
n_init_sample = 10
benchmarkfunctions = {
                #"schwefel":benchmarks.schwefel,
                #"ackley":benchmarks.himmelblau,
                #"rastrigin":benchmarks.rastrigin,
                #"bohachevsky":benchmarks.bohachevsky,
                #"schaffer":benchmarks.schaffer,
                "himmelblau":benchmarks.himmelblau
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
        df = pd.DataFrame([], columns=['run{}'.format(_+1) for _ in range(runs)])
        df.to_csv(csv_name, mode='w', header=True, index=False)
        
        optimizer = create_optimizer(dim, fitness, lb, ub, n_step, n_init_sample)
        
        for n in range(n_step):
            xopt, fopt, A, B, C = optimizer.step()
            comm.Barrier()
            
            # gather running results
            __ = comm.gather(fopt, root=0)
            
            if rank == 0:
                y_hist_best[n, :] = __
                mean_ = np.mean(__)
                error_ = np.std(__, ddof=1) / np.sqrt(runs)
                print 'step {}:'.format(n + 1) 
                print 'mean : {}, std error: {}'.format(mean_, error_)
                print
                
                # append the new data the csv
                df = pd.DataFrame(np.atleast_2d(__))
                df.to_csv(csv_name, mode='a', header=False, index=False)