#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:57:47 2017

@author: wangronin
"""

import pdb
from sklearn.svm import SVR
import numpy as np
from owck import GaussianProcess_extra as GaussianProcess

np.random.seed(1)
from fitness import sphere

dim = 2
# test problem: to fit a so-called Rastrigin function in 20D
X = np.random.rand(20, dim)
y = sphere(X.T)

print y.std() ** 2

lb = np.array([0] * dim)
ub = np.array([1] * dim)

thetaL = 1e-5 * (ub - lb) * np.ones(dim)
thetaU = 10 * (ub - lb) * np.ones(dim)
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

gp = GaussianProcess(regr='constant', corr='matern',
                 theta0=theta0, thetaL=thetaL,
                 thetaU=thetaU, nugget=None,
                 optimizer='BFGS',
                 nugget_estim=True, normalize=True,
                 verbose=False, random_start=10,
                 random_state=None)

gp.fit(X, y)

print y
print gp.predict(X)

print gp.theta_
print gp.sigma2

pdb.set_trace()