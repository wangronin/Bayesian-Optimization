from pdb import set_trace

import numpy as np
import sys, os
import pandas as pd
sys.path.insert(0, '../')

from bayes_optim import NarrowingBO, ContinuousSpace
from bayes_optim.Surrogate import GaussianProcess, trend

np.random.seed(123)
dim = 10
subdim = 5
lb, ub = 1, 5


def fitness(x):
    x = np.asarray(x[:subdim])
    return np.sum([x[1] ** x[0] for x in enumerate(x)])


def corr_fsel(data, model, active_fs):
    """ Pearson correlation-based feature selection
    Considering the points evaluated and the active features,
    the correlation is calculated. Then, the feature with the smallest
    correlation is discarded. 
    """
    if len(active_fs) == 1:
        return {}
    df = data.to_dataframe()
    df = df[active_fs + ['f']]
    cor = df.corr()
    cor_fitness = abs(cor['f'])
    fs = cor_fitness.sort_values(ascending=True).index[0]
    # TODO set the value for the discarded feature
    return {fs: 0}


def mean_improvement(data, model, metrics):
    """ Mean fitness improvement criteria.
    The mean fitness is calculated for all the available data
    points. If the mean is improving (ge) in relation to the
    previous one (i.e., metrics in the input). Then, a True value
    is returned, along with the new calculated mean.
    """
    _mean = np.mean([x.fitness for x in data])
    if (('mean' in metrics and _mean >= metrics['mean']) or
            'mean' not in metrics):
        return True, {'mean': _mean}
    return False, {'mean': _mean}


space = ContinuousSpace([lb, ub]) * dim

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
    optimizer='BFGS',
    wait_iter=3,
    random_start=dim,
    likelihood='concentrated',
    eval_budget=100 * dim)

opt = NarrowingBO(
    search_space=space, 
    obj_fun=fitness, 
    model=model, 
    DoE_size=5,
    max_FEs=25, 
    verbose=True,
    n_point=1,
    minimize=False,
    narrowing_fun=corr_fsel,
    narrowing_improving_fun=mean_improvement,
    narrowing_FEs=5)

print(opt.run())
