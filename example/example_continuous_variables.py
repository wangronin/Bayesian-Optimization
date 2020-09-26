from pdb import set_trace

import numpy as np
import sys, os
sys.path.insert(0, '../')

from bayes_optim import BO, ContinuousSpace
from bayes_optim.Surrogate import GaussianProcess, trend

np.random.seed(123)
dim = 5
lb, ub = -1, 5

def fitness(x):
    x = np.asarray(x)
    return np.sum(x ** 2)

space = ContinuousSpace([lb, ub]) * dim

mean = trend.constant_trend(dim, beta=None)
thetaL = 1e-10 * (ub - lb) * np.ones(dim)
thetaU = 10 * (ub - lb) * np.ones(dim)
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

model = GaussianProcess(
    mean=mean, corr='squared_exponential',
    theta0=theta0, thetaL=thetaL, thetaU=thetaU,
    nugget=0, noise_estim=False,
    optimizer='BFGS', wait_iter=3, random_start=dim,
    likelihood='concentrated', eval_budget=100 * dim
)

opt = BO(
    search_space=space, 
    obj_fun=fitness, 
    model=model, 
    DoE_size=5,
    max_FEs=50, 
    verbose=True, 
    n_point=1
)
print(opt.run())
