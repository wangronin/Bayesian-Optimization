import sys

import numpy as np
import pytest

sys.path.insert(0, "../")
from bayes_optim import MOBO
from bayes_optim._exception import RecommendationUnavailableError
from bayes_optim.search_space import (BoolSpace, DiscreteSpace, IntegerSpace,
                                      RealSpace)
from bayes_optim.surrogate import GaussianProcess, RandomForest

np.random.seed(123)


def f1(x):
    return (
        (x["continuous"] - 3) ** 2
        + abs(x["ordinal"] + 10) / 123.0
        + (x["nominal"] != "OK") * 2
        + int(x["bool"]) * 3
    )


def f2(x):
    return (
        x["continuous"] ** 3
        + abs(x["ordinal"] - 10) / 123.0
        + (x["nominal"] != "G") * 2
        + int(not x["bool"]) * 3
    )


def test_3D():
    search_space = (
        RealSpace([0, 100], var_name="Kp", precision=2)
        + RealSpace([0, 100], var_name="Ki", precision=2)
        + RealSpace([0, 100], var_name="Kd", precision=2)
    )
    f1 = lambda x: x["Kp"] ** 2 + x["Ki"] + x["Kd"] ** 2
    f2 = lambda x: x["Kp"] + x["Ki"] ** 2 + x["Kd"] ** 2
    f3 = lambda x: x["Kp"] ** 2 + x["Ki"] + x["Kd"]
    dim = search_space.dim
    thetaL = 1e-10 * 100 * np.ones(dim)
    thetaU = 10 * 100 * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

    model = GaussianProcess(
        theta0=theta0,
        thetaL=thetaL,
        thetaU=thetaU,
        nugget=0,
        noise_estim=False,
        likelihood="concentrated",
    )
    opt = MOBO(
        search_space=search_space,
        obj_fun=(f1, f2, f3),
        model=model,
        max_FEs=100,
        DoE_size=1,  # the initial DoE size
        eval_type="dict",
        n_job=1,  # number of processes
        verbose=True,  # turn this off, if you prefer no output
        minimize=True,
        acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
    )
    for _ in range(100):
        X = opt.ask(1)
        opt.tell(X, [(f1(x), f2(x), f3(x)) for x in X])

    opt.recommend()
    with pytest.raises(NotImplementedError):
        X = opt.ask(3)


def test_with_constraints():
    search_space = (
        RealSpace([0, 100], var_name="left", precision=2)
        + RealSpace([0, 100], var_name="up", precision=2)
        + RealSpace([0, 100], var_name="right", precision=2)
    )
    f1 = lambda x: x["left"] ** 2 + x["up"] + x["right"] ** 2
    f2 = lambda x: 10 * x["left"] - x["up"] ** 2 + x["right"]
    g = lambda x: x["left"] + x["up"] + x["right"] - 100
    dim = search_space.dim
    thetaL = 1e-10 * 100 * np.ones(dim)
    thetaU = 10 * 100 * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

    model = GaussianProcess(
        theta0=theta0,
        thetaL=thetaL,
        thetaU=thetaU,
        nugget=0,
        noise_estim=False,
        likelihood="concentrated",
    )
    opt = MOBO(
        search_space=search_space,
        obj_fun=(f1, f2),
        ineq_fun=g,
        model=model,
        max_FEs=100,
        DoE_size=1,  # the initial DoE size
        eval_type="dict",
        n_job=1,  # number of processes
        verbose=True,  # turn this off, if you prefer no output
        acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
    )
    for _ in range(50):
        X = opt.ask(1)
        opt.tell(X, [(f1(x), f2(x)) for x in X])
        assert g(X[0]) <= 0

    opt.recommend()
    with pytest.raises(NotImplementedError):
        X = opt.ask(3)


def test_fixed_var():
    search_space = (
        BoolSpace(var_name="bool")
        + IntegerSpace([5, 15], var_name="ordinal")
        + RealSpace([-5, 5], var_name="continuous", precision=2)
        + DiscreteSpace(["OK", "A", "B", "C", "D", "E", "F", "G"], var_name="nominal")
    )
    opt = MOBO(
        search_space=search_space,
        obj_fun=(f1, f2),
        model=RandomForest(levels=search_space.levels),
        max_FEs=100,
        DoE_size=1,  # the initial DoE size
        eval_type="dict",
        n_job=1,  # number of processes
        verbose=True,  # turn this off, if you prefer no output
    )
    X = opt.ask(3, fixed={"ordinal": 5, "continuous": 3.2})
    assert all([x["ordinal"] == 5 and x["continuous"] == 3.2 for x in X])
    opt.tell(X, [(f1(x), f2(x)) for x in X])
    X = opt.ask(1, fixed={"nominal": "OK", "bool": False})
    assert all([x["nominal"] == "OK" and not x["bool"] for x in X])
    opt.recommend()

    with pytest.raises(NotImplementedError):
        X = opt.ask(3)


def test_recommend():
    search_space = (
        RealSpace([10, 30], var_name="p1", precision=2)
        + IntegerSpace([20, 40], var_name="p2")
        + DiscreteSpace([128, 256, 512], var_name="p3")
        + BoolSpace(var_name="p4")
    )
    opt = MOBO(
        search_space=search_space,
        obj_fun=(f1, f2),
        model=RandomForest(levels=search_space.levels),
        max_FEs=100,
        DoE_size=3,  # the initial DoE size
        eval_type="dict",
        n_job=1,  # number of processes
        verbose=True,  # turn this off, if you prefer no output
    )
    with pytest.raises(RecommendationUnavailableError):
        opt.recommend()


def test_constraint():
    search_space = (
        BoolSpace(var_name="bool")
        + IntegerSpace([5, 15], var_name="ordinal")
        + RealSpace([-5, 5], var_name="continuous")
        + DiscreteSpace(["OK", "A", "B", "C", "D", "E", "F", "G"], var_name="nominal")
    )
    opt = MOBO(
        search_space=search_space,
        obj_fun=(f1, f2),
        model=RandomForest(levels=search_space.levels),
        ineq_fun=lambda x: x["continuous"],
        max_FEs=10,
        DoE_size=3,  # the initial DoE size
        eval_type="dict",
        n_job=1,  # number of processes
        verbose=True,  # turn this off, if you prefer no output
    )
    opt.run()
