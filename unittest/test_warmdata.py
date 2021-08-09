import sys

import numpy as np

sys.path.insert(0, "../")

from bayes_optim import BO, DiscreteSpace, IntegerSpace, RealSpace
from bayes_optim.surrogate import GaussianProcess, RandomForest

np.random.seed(42)


def obj_fun(x):
    x_r, x_i, x_d = np.array(x[:2]), x[2], x[3]
    if x_d == "OK":
        tmp = 0
    else:
        tmp = 1
    return np.sum((x_r + np.array([2, 2])) ** 2) + abs(x_i - 10) * 10 + tmp


def test_warm_data_with_GPR():
    dim = 2
    lb, ub = -5, 5

    def fitness(x):
        x = np.asarray(x)
        return np.sum(x ** 2)

    X = np.random.rand(5, dim) * (ub - lb) + lb
    y = [fitness(x) for x in X]
    space = RealSpace([lb, ub]) * dim

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
    opt = BO(
        search_space=space,
        obj_fun=fitness,
        model=model,
        warm_data=(X, y),
        max_FEs=10,
        verbose=True,
        n_point=1,
    )
    assert np.all(np.asarray(opt.data) == np.asarray(opt.warm_data))
    assert opt.model.is_fitted
    opt.run()


def test_warm_data_with_RF():
    space = (
        RealSpace([-10, 10]) * 2
        + IntegerSpace([5, 15])
        + DiscreteSpace(["OK", "A", "B", "C", "D", "E", "F", "G"])
    )

    X = space.sample(10)
    y = [obj_fun(x) for x in X]

    model = RandomForest(levels=space.levels)
    opt = BO(
        search_space=space,
        obj_fun=obj_fun,
        model=model,
        minimize=True,
        eval_type="list",
        max_FEs=5,
        verbose=True,
        acquisition_fun="EI",
        warm_data=(X, y),
    )
    opt.run()
    assert opt.data.shape[0] == 15
