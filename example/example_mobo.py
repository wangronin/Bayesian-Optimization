import sys

import numpy as np

sys.path.insert(0, "../")

from bayes_optim import RealSpace
from bayes_optim.mobo import MOBO
from bayes_optim.surrogate import GaussianProcess, RandomForest, trend

np.random.seed(123)
dim = 2
lb, ub = -10, 10


def f1(x):
    x = np.asarray(x)
    return np.sum((x + 5) ** 2) * 1000


def f2(x):
    x = np.asarray(x)
    return np.sum((x - 5) ** 2)


space = RealSpace([lb, ub]) * dim
# Bayesian optimization also uses a Surrogate model
# For mixed variable type, the random forest is typically used
model = RandomForest()

mean = trend.constant_trend(dim, beta=None)
thetaL = 1e-10 * (ub - lb) * np.ones(dim)
thetaU = 10 * (ub - lb) * np.ones(dim)
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

model = GaussianProcess(
    theta0=theta0,
    thetaL=thetaL,
    thetaU=thetaU,
    nugget=1e-8,
    noise_estim=False,
    wait_iter=3,
    random_start=dim,
    eval_budget=100 * dim,
)

opt = MOBO(
    search_space=space,
    obj_fun=(f1, f2),
    model=model,
    DoE_size=5,
    max_FEs=20,
    verbose=True,
    # n_point=3,
    acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
    # acquisition_optimization={"optimizer": "MIES"},
)
print(opt.run())
