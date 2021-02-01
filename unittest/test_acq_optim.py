import sys
import numpy as np
sys.path.insert(0, '../')
from bayes_optim.acquisition_optim import OnePlusOne_CMA, OnePlusOne_Cholesky_CMA

def obj_fun_min(x):
    return np.sum(x ** 2)

def obj_fun_max(x):
    return -np.sum(x ** 2)

def h(x):
    return np.sum(x) - 1

def g(x):
    return 1 - np.sum(x)

def test_OnePlusOne_Cholesky_CMA():
    _, __, stop_dict = OnePlusOne_Cholesky_CMA(
        dim=2, obj_fun=obj_fun_min, lb=-5, ub=5, sigma0=2, ftol=1e-2,
        verbose=False, random_seed=42
    ).run()
    assert 'ftol' in stop_dict


def test_OnePlusOne_Cholesky_CMA_constraint():
    xopt, _, __, = OnePlusOne_Cholesky_CMA(
        dim=2, obj_fun=obj_fun_max, h=h, lb=-5, ub=5, sigma0=2, max_FEs=500,
        minimize=False, verbose=False, random_seed=42
    ).run()
    assert np.isclose(h(xopt), 0, atol=1e-2)


    OnePlusOne_Cholesky_CMA(
        dim=2, obj_fun=obj_fun_min, g=g, lb=-5, ub=5, sigma0=2, max_FEs=1000,
        verbose=True, random_seed=42
    ).run()
    assert np.isclose(g(xopt), 0, atol=1e-2)
