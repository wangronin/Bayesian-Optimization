import sys

import numpy as np
import pytest

sys.path.insert(0, "../")
from bayes_optim import BO, IntegerSpace, RealSpace
from bayes_optim._exception import ConstraintEvaluationError
from bayes_optim.search_space import DiscreteSpace
from bayes_optim.surrogate import GaussianProcess, RandomForest


def obj_fun2(x):
    return (x["pc"] - 0.2) ** 2 + x["mu"] + x["lambda"] + np.abs(x["p"] - 0.7)


def obj_fun(x):
    return np.sum(np.array(x) ** 2) + 5 * np.sum(np.array(x)) + 10


def h(x):
    return np.sum(x) - 1


def g(x):
    return [-x["pc"], x["mu"] - 1.9]


def test_BO_equality():
    dim = 2
    search_space = RealSpace([0, 1]) * dim
    thetaL = 1e-5 * np.ones(dim)
    thetaU = np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL
    model = GaussianProcess(
        corr="squared_exponential",
        theta0=theta0,
        thetaL=thetaL,
        thetaU=thetaU,
        nugget=1e-1,
        random_state=42,
    )
    xopt, _, __ = BO(
        search_space=search_space,
        obj_fun=obj_fun,
        eq_fun=h,
        model=model,
        max_FEs=20,
        DoE_size=3,
        acquisition_fun="MGFI",
        acquisition_par={"t": 2},
        acquisition_optimization={"optimizer": "BFGS"},
        verbose=True,
        random_seed=42,
    ).run()
    assert np.isclose(h(xopt), 0, atol=1e-1)


def test_BO_constraints():
    search_space = (
        IntegerSpace([1, 10], var_name="mu")
        + IntegerSpace([1, 10], var_name="lambda")
        + RealSpace([0, 1], var_name="pc")
        + RealSpace([0.005, 0.5], var_name="p")
    )
    model = RandomForest(levels=search_space.levels)
    xopt, _, __ = BO(
        search_space=search_space,
        obj_fun=obj_fun2,
        ineq_fun=g,
        model=model,
        max_FEs=10,
        DoE_size=3,
        eval_type="dict",
        acquisition_fun="MGFI",
        acquisition_par={"t": 2},
        n_job=1,
        n_point=1,
        verbose=True,
        random_seed=42,
    ).run()
    assert isinstance(xopt, dict)
    assert all(np.array(g(xopt)) <= 0)


def test_BO_bad_constraints():
    search_space = (
        DiscreteSpace(["1", "2", "3"], var_name="lambda")
        + RealSpace([0, 1], var_name="pc")
        + RealSpace([0.005, 0.5], var_name="p")
    )
    model = RandomForest(levels=search_space.levels)
    with pytest.raises(ConstraintEvaluationError):
        BO(
            search_space=search_space,
            obj_fun=lambda x: 10 * (x[0] == "3") + x[1] * x[2],
            ineq_fun=lambda x: sum(np.array(x) ** 2),
            model=model,
            max_FEs=10,
            DoE_size=3,
            eval_type="list",
            acquisition_fun="MGFI",
            acquisition_par={"t": 2},
            n_job=1,
            n_point=1,
            verbose=True,
            random_seed=42,
        ).run()
