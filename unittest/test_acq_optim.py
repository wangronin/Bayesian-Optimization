import sys

import numpy as np

sys.path.insert(0, "../")
from bayes_optim.acquisition_optim import MIES, OnePlusOne_Cholesky_CMA
from bayes_optim.search_space import Bool, Discrete, Integer, Ordinal, Real, RealSpace, SearchSpace


def obj_fun_min(x):
    return np.sum(x ** 2)


def obj_fun_max(x):
    return -np.sum(x ** 2)


def obj_fun_mies(x):
    x1, x2, x3, x4, x5 = x.tolist()
    return x1 ** 2 + abs(x5) + x4 - 2 + 10 * int(x3 == "B") + 5 * int(not x2)


def h(x):
    return np.sum(x) - 1


def g(x):
    return 1 - np.sum(x)


def test_OnePlusOne_Cholesky_CMA():
    _, __, stop_dict = OnePlusOne_Cholesky_CMA(
        search_space=RealSpace([-5, 5]) * 2,
        obj_fun=obj_fun_min,
        sigma0=2,
        ftol=1e-2,
        verbose=False,
        random_seed=42,
    ).run()
    assert "ftol" in stop_dict


def test_OnePlusOne_Cholesky_CMA_constraint():
    xopt, _, __, = OnePlusOne_Cholesky_CMA(
        search_space=RealSpace([-5, 5]) * 2,
        obj_fun=obj_fun_max,
        h=h,
        sigma0=2,
        max_FEs=500,
        minimize=False,
        verbose=False,
        random_seed=42,
    ).run()
    assert np.isclose(h(xopt), 0, atol=1e-1)

    OnePlusOne_Cholesky_CMA(
        search_space=RealSpace([-5, 5]) * 2,
        obj_fun=obj_fun_min,
        g=g,
        sigma0=2,
        max_FEs=1000,
        verbose=False,
        random_seed=42,
    ).run()
    assert np.isclose(g(xopt), 0, atol=1e-1)


def test_MIES():
    space = SearchSpace(
        [
            Real([0, 1], "x1"),
            Bool("x2"),
            Discrete(["A", "B", "C"], "x3"),
            Ordinal([0, 2, 4, 6, 8], "x4"),
            Integer([0, 10], "x5"),
        ]
    )
    _, __, stop_dict = MIES(
        search_space=space,
        max_eval=100,
        obj_func=obj_fun_mies,
        verbose=False,
        eval_type="list",
        minimize=True,
    ).optimize()
    assert "max_eval" in stop_dict
