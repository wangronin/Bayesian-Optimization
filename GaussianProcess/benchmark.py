#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:01:39 2017

@author: wangronin
"""

import pdb
from copy import deepcopy
from GaussianProcess.trend import constant_trend
from GaussianProcess import GaussianProcess as GaussianProcess

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from deap import benchmarks
import numpy as np
from numpy.random import randn

np.random.seed(123)
plt.ioff()
fig_width = 21.5
fig_height = fig_width / 3.2

def fitness(X):
    X = np.atleast_2d(X)
    return np.array([benchmarks.rastrigin(x)[0] for x in X]) \
        + np.sqrt(noise_var) * randn(X.shape[0])

dim = 2
n_init_sample = 100
noise_var = 0
n_run = 30

x_lb = np.array([-5] * dim)
x_ub = np.array([5] * dim)

length_lb = [1e-10] * dim
length_ub = [1e3] * dim

thetaL = 1e-5 * (x_ub - x_lb) * np.ones(dim)
thetaU = 10 * (x_ub - x_lb) * np.ones(dim)

# initial search point for hyper-parameters
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

mean = constant_trend(dim, beta=None)
models = [GaussianProcess(mean=mean, corr='matern', theta0=theta0, thetaL=thetaL, thetaU=thetaU,
                           nugget=None, nugget_estim=False, optimizer='CMA', verbose=True,
                           wait_iter=3, random_start=30, eval_budget=1e3, 
                           likelihood='concentrated'),
           GaussianProcess(mean=mean, corr='matern', theta0=theta0, thetaL=thetaL, thetaU=thetaU,
                           nugget=None, nugget_estim=False, optimizer='CMA', verbose=True,
                           wait_iter=3, random_start=30, eval_budget=1e3,
                           likelihood='restricted')]

r2 = np.zeros((n_run, len(models)))
for i in range(n_run):
    X = np.random.rand(n_init_sample, dim) * (x_ub - x_lb) + x_lb
    y = fitness(X)
    
    X_test = np.random.rand(int(1e5), dim) * (x_ub - x_lb) + x_lb
    y_test = fitness(X_test)
    
    print 'run {}/{}'.format(i+1, n_run)
    for k, model_ in enumerate(models):
        model = deepcopy(model_)
        model.fit(X, y)
        
        y_hat = model.predict(X_test)
        r2[i, k] = r2_score(y_test, y_hat)
        print

print np.median(r2, axis=0)

fig0, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), subplot_kw={'aspect': 'auto'}, 
                        dpi=100)

ax.boxplot(r2, 0, 'gD')

plt.tight_layout()
plt.show()

pdb.set_trace()
