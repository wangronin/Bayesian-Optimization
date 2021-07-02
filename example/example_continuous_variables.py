import os
import sys

import numpy as np

sys.path.insert(0, "./")

from bayes_optim import BO, RealSpace
from bayes_optim.surrogate import GaussianProcess, trend

np.random.seed(123)
dim = 5
lb, ub = -1, 5


def fitness(x):
    x = np.asarray(x)
    return np.sum(x ** 2)


space = RealSpace([lb, ub]) * dim
thetaL = 1e-10 * (ub - lb) * np.ones(dim)
thetaU = 10 * (ub - lb) * np.ones(dim)
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

model = GaussianProcess(
    theta0=theta0,
    thetaL=thetaL,
    thetaU=thetaU,
    nugget=1e-3,
    noise_estim=True,
    optimizer="BFGS",
    wait_iter=3,
    random_start=dim,
    likelihood="concentrated",
    eval_budget=100 * dim,
)

opt = BO(
    search_space=space,
    obj_fun=fitness,
    model=model,
    DoE_size=5,
    max_FEs=20,
    verbose=True,
    n_point=1,
    acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
)
print(opt.run())
