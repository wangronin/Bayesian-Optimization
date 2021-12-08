import functools
import os
import sys
from typing import List, Union

import numpy as np
import pandas as pd

sys.path.insert(0, "../")

from bayes_optim import NarrowingBO, RealSpace
from bayes_optim.surrogate import GaussianProcess, trend

np.random.seed(3108)
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
    df = df[list(active_fs) + ["f"]]
    cor = df.corr()
    cor_fitness = abs(cor["f"])
    # TODO is the name of the variable influencing the sort?
    fs = cor_fitness.sort_values(ascending=True).index[0]
    # TODO set the value for the discarded feature
    return fs, 0


def mean_improvement(data, model, metrics, tolerance=0.00, minimize=True):
    """Mean fitness improvement criteria.
    The mean fitness is calculated for all the available data
    points. If the mean is improving (ge) in relation to the
    previous one (i.e., metrics in the input). Then, a True value
    is returned, along with the new calculated mean.
    """
    _mean = np.mean([x.fitness for x in data])
    improving = False
    if "mean" not in metrics:
        improving = True
    else:
        diff = (_mean - metrics["mean"]) / metrics["mean"]
        diff = - diff if minimize else diff
        improving = True if diff >= -tolerance else False

    return improving, {"mean": _mean}


def min_mean_improvement(data, model, metrics):
    return mean_improvement(data, model, metrics, tolerance=0.00, minimize=True)


def max_mean_improvement(data, model, metrics):
    return mean_improvement(data, model, metrics, tolerance=0.00, minimize=False)



space = RealSpace([lb, ub]) * dim

model = GaussianProcess(domain=space, n_restarts_optimizer=dim)

opt = NarrowingBO(
    search_space=space,
    obj_fun=branin,
    model=model,
    DoE_size=5,
    max_FEs=50,
    verbose=True,
    n_point=1,
    minimize=True,
    var_selector=corr_fsel,
    search_space_improving_fun=min_mean_improvement,
    var_selection_FEs=10
)

print(opt.run())
