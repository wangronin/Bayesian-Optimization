import numpy as np
import sys
sys.path.insert(0, '../')

from bayes_optim.acquisition_optim import OnePlusOne_CMA, OnePlusOne_Cholesky_CMA

def obj_fun(x):
    return np.sum(x ** 2)

def h(x):
    return np.sum(x) - 1

def g(x):
    return 1 - np.sum(x)

def test_OnePlusOne_Cholesky_CMA():
    OnePlusOne_Cholesky_CMA(
        2, obj_fun, lb=-5, ub=5, sigma0=0.2, max_FEs=100, verbose=False
    ).run()

    OnePlusOne_Cholesky_CMA(
        dim=2, obj_fun=obj_fun, h=h, lb=-5, ub=5, sigma0=2, max_FEs=500, verbose=False
    ).run()

    OnePlusOne_Cholesky_CMA(
        dim=2, obj_fun=obj_fun, g=g, lb=-5, ub=5, sigma0=2, max_FEs=500, verbose=False
    ).run()

    np.random.seed(42)
    _, __, stop_dict = OnePlusOne_Cholesky_CMA(
        dim=2, obj_fun=obj_fun, lb=-5, ub=5, sigma0=2, ftol=1e-2, verbose=False
    ).run()
    assert 'ftol' in stop_dict
