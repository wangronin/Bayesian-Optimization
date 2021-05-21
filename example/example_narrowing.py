import functools
import os
import sys
from typing import List, Union

import numpy as np
import pandas as pd

sys.path.insert(0, "../")

from bayes_optim import NarrowingBO, RealSpace
from bayes_optim.surrogate import GaussianProcess, trend

np.random.seed(123)
dim = 25 # Should be greater than 2
lb, ub = -5, 15


def specifiy_dummy_vars(d_eff: int):
    def wrapper(func):
        @functools.wraps(func)
        def inner(x):
            x = np.asarray(x[:d_eff])
            return func(x)
        return inner
    return wrapper


@specifiy_dummy_vars(2)
def branin(x):
    """ Branin function (https://www.sfu.ca/~ssurjano/branin.html)
    Global minimum 0.397881 at (-Pi, 12.275), (Pi, 2.275), and (9.42478, 2.475)
    """
    x1 = x[0]
    x2 = x[1]
    g_x = (x2 - (5.1 * x1**2)/(4 * np.pi**2 ) + 5 * x1 / np.pi  -6) ** 2 + 10 * (1 - (1 / (8 * np.pi))) * np.cos(x1) + 10
    return np.abs(0.397881 - g_x)


@specifiy_dummy_vars(2)
def fitness(x):
    return np.sum([x[1] ** x[0] for x in enumerate(x)])


def corr_fsel(data, model, active_fs):
    """Pearson correlation-based feature selection
    Considering the points evaluated and the active features,
    the correlation is calculated. Then, the feature with the smallest
    correlation is discarded.
    """
    if len(active_fs) == 1:
        return {}
    df = pd.DataFrame(data.tolist(), columns=data.var_name.tolist())
    df["f"] = data.fitness
    df = df[active_fs + ["f"]]
    cor = df.corr()
    cor_fitness = abs(cor["f"])
    # TODO is the name of the variable influencing the sort?
    fs = cor_fitness.sort_values(ascending=True).index[0]
    # TODO set the value for the discarded feature
    return {fs: 0}


def mean_improvement(data, model, metrics):
    """Mean fitness improvement criteria.
    The mean fitness is calculated for all the available data
    points. If the mean is improving (ge) in relation to the
    previous one (i.e., metrics in the input). Then, a True value
    is returned, along with the new calculated mean.
    """
    _mean = np.mean([x.fitness for x in data])
    if ("mean" in metrics and _mean >= metrics["mean"]) or "mean" not in metrics:
        return True, {"mean": _mean}
    return False, {"mean": _mean}


space = RealSpace([lb, ub]) * dim
mean = trend.constant_trend(dim, beta=None)
thetaL = 1e-10 * (ub - lb) * np.ones(dim)
thetaU = 10 * (ub - lb) * np.ones(dim)
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

model = GaussianProcess(
    theta0=theta0,
    thetaL=thetaL,
    thetaU=thetaU,
    nugget=0,
    noise_estim=False,
    optimizer="BFGS",
    wait_iter=3,
    random_start=dim,
    likelihood="concentrated",
    eval_budget=100 * dim,
)

opt = NarrowingBO(
    search_space=space,
    obj_fun=branin,
    model=model,
    DoE_size=30,
    max_FEs=500,
    verbose=True,
    n_point=1,
    minimize=True,
    narrowing_fun=corr_fsel,
    narrowing_improving_fun=mean_improvement,
    narrowing_FEs=5,
)

print(opt.run())
