from pdb import set_trace

import numpy as np
import sys, os
sys.path.insert(0, '../')

from pyDOE import lhs
from GaussianProcess import GaussianProcess
from GaussianProcess.trend import constant_trend

dim = 5
lb, ub = -5, 5

def fitness(x):
    x = np.asarray(x)
    return np.sum(x ** 2, axis=1)

x_lb = np.array([lb] * dim)
x_ub = np.array([ub] * dim)

N = 5
X = lhs(dim, samples=10, criterion='cm') * (x_ub - x_lb) + x_lb
# X = np.random.rand(5, dim) * (x_ub - x_lb) + x_lb
y = fitness(X)

y =  (y - np.min(y)) / (np.max(y) - np.min(y)) + 10
# y = (y - np.mean(y)) / np.std(y)

mean = constant_trend(dim, beta=None)
thetaL = 1e-10 * (ub - lb) * np.ones(dim)
thetaU = 2 * (ub - lb) * np.ones(dim)
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

model = GaussianProcess(
    mean=mean, corr='squared_exponential',
    theta0=theta0, thetaL=thetaL, thetaU=thetaU,
    nugget=0, noise_estim=True,
    optimizer='BFGS', wait_iter=3, random_start=dim,
    likelihood='concentrated', eval_budget=100 * dim,
    verbose=True
)

model.fit(X, y.ravel())
