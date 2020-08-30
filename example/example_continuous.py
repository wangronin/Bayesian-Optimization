from pdb import set_trace

import numpy as np
import sys, os

sys.path.insert(0, '../')

from BayesOpt import AnnealingBO, BO, ContinuousSpace, OrdinalSpace, \
    NominalSpace, RandomForest

from GaussianProcess import GaussianProcess
from GaussianProcess.trend import constant_trend

np.random.seed(123)

dim = 5
lb, ub = -1, 5

def fitness(x):
    x = np.asarray(x)
    return np.sum(x ** 2)

space = ContinuousSpace([lb, ub]) * dim

mean = constant_trend(dim, beta=None)
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

def mixed_integer():
    def fitness(x):
        x_r, x_i, x_d = np.array(x[:2]), x[2], x[3]
        if x_d == 'OK':
            tmp = 0
        else:
            tmp = 1
        return np.sum(x_r ** 2) + abs(x_i - 10) / 123. + tmp * 2

    space = (ContinuousSpace([-5, 5]) * 2) + \
        OrdinalSpace([5, 15]) + \
        NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])

    levels = space.levels if hasattr(space, 'levels') else None
    model = RandomForest(levels=levels)

    opt = AnnealingBO(
        search_space=space, 
        obj_fun=fitness, 
        model=model, 
        max_FEs=300, 
        verbose=True, 
        n_job=3, 
        n_point=3,
        acquisition_fun='MGFI',
        acquisition_par={'t' : 2},
        DoE_size=3
    )
    opt.run()
