import numpy as np
from bayes_optim import MOBO
from bayes_optim.search_space import BoolSpace, DiscreteSpace, IntegerSpace, RealSpace
from bayes_optim.surrogate import RandomForest

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


def test_fixed_var():
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

    try:
        X = opt.ask(3)
    except NotImplementedError:
        pass


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
    assert opt.recommend() is None
