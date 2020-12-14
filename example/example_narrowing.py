from pdb import set_trace

import numpy as np
import sys, os
sys.path.insert(0, '../')

from bayes_optim import NarrowingBO, ContinuousSpace
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
    theta0=theta0, thetaL=thetaL, thetaU=thetaU,
    nugget=0, noise_estim=False,
    optimizer='BFGS', wait_iter=3, random_start=dim,
    likelihood='concentrated', eval_budget=100 * dim
)

def fsel(data, model, active_fs):
    if len(active_fs) >= 1:
        return {active_fs[-1]: 0}
    return {}

def narrowing_improving_fun(data, model, metrics):
    if len(data) > 19:
        return False, {}
    return True, {}

opt = NarrowingBO(
    search_space=space, 
    obj_fun=fitness, 
    model=model, 
    DoE_size=5,
    max_FEs=25, 
    verbose=True,
    n_point=1,
    narrowing_fun=fsel,
    narrowing_improving_fun=narrowing_improving_fun,
    narrowing_FEs=5
)
print(opt.run())
