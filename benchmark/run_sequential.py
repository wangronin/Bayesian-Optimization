#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runs an entire experiment for benchmarking Bayesian Optimization on a testbed.

"""

import pdb
import time
import numpy as np
from numpy import floor, log

import fgeneric
import bbobbenchmarks as bn

from BayesOpt.BayesOptNew import BayesOpt
#from BayesOpt import BayesOpt
from BayesOpt.SearchSpace import ContinuousSpace

from GaussianProcess import GaussianProcess
from GaussianProcess.trend import constant_trend

datapath = './bbob_data/test'
opts = dict(algid='Bayesopt', comments='Bayesian Optimization')
maxfunevals = '50 * dim'  

infill = 'MGFI'
schedule = 'None'
                        

def run_bayesopt(obj_fun, dim, maxfunevals, ftarget=-np.Inf):
    n_init_sample = 10 * dim
    search_space = ContinuousSpace([-4, 4]) * dim
    
    thetaL = 1e-10 * 8 * np.ones(dim)
    thetaU = 10 * 8 * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL
    
    mean = constant_trend(dim, beta=None)
    model = GaussianProcess(mean=mean, corr='matern',
                            theta0=theta0, thetaL=thetaL, thetaU=thetaU,
                            nugget=1e-6, noise_estim=False, verbose=False,
                            optimizer='BFGS', wait_iter=3, random_start=min(2 * dim, 10),
                            likelihood='concentrated', eval_budget=int(8 + floor(40 * log(dim))))
                            
    opt = BayesOpt(search_space, obj_fun, model, random_seed=None, 
                   infill=infill, t0=2, schedule=schedule,
                   n_init_sample=n_init_sample, max_eval=maxfunevals, minimize=True, 
                   max_infill_eval=50 * dim, n_restart=min(3 * dim, 20), verbose=True, 
                   optimizer='BFGS')

    opt.run()
    

t0 = time.time()
np.random.seed(666)

f = fgeneric.LoggingFunction(datapath, **opts)

n_instance = 15
error = np.zeros(n_instance)

for dim in (2,):  
    for f_name in bn.nfreeIDs: # or bn.noisyIDs
        for i, iinstance in enumerate(range(n_instance)):
            f.setfun(*bn.instantiate(f_name, iinstance=iinstance))

            run_bayesopt(f.evalfun, dim,  eval(maxfunevals), f.ftarget)
                
            f.finalizerun()
            print('  f%d in %d-D, instance %d: FEs=%d, '
                  'fbest-ftarget=%.4e, elapsed time [m]: %.3f'
                  % (f_name, dim, iinstance, f.evaluations, 
                     f.fbest - f.ftarget, (time.time() - t0) / 60.))
            error[i] = f.fbest - f.ftarget
            
        print '      median fitness error runs: %f' % np.mean(error)
        print '      date and time: %s' % (time.asctime())
    print '---- dimension %d-D done ----' % dim

