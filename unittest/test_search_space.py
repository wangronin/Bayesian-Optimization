import json
import re
import sys

sys.path.insert(0, "../")

import numpy as np
import pytest
from bayes_optim.search_space import (
    Discrete,
    DiscreteSpace,
    Integer,
    IntegerSpace,
    Node,
    Real,
    RealSpace,
    SearchSpace,
    Subset,
    SubsetSpace,
    Variable,
)
from bayes_optim.solution import Solution


def test_Variable():
    x = Real([1e-20, 5], "x", 2, scale="log10")
    bounds_transformed = getattr(x, "_bounds_transformed")
    assert bounds_transformed[0] == -20
    assert np.isclose(bounds_transformed[1], 0.6989700043360189)

    x = IntegerSpace([[5, 10]] * 3, ["x", "y", "z"])
    assert all(np.asarray(x.var_name) == np.asarray(["x", "y", "z"]))
    assert all(np.asarray(x.var_type) == np.asarray(["Integer"] * 3))


def test_real_warning():
    x = Real([-np.inf, 5], "x", 2, scale="log10")
    assert x.bounds[0] == 1e-300


def test_SearchSpace_var_name():
    cs = (
        RealSpace([1e-10, 1e-1], "x", 1e-3, scale="log")
        + IntegerSpace([-10, 10], "y")
        + DiscreteSpace(["A", "B", "C", "D", "E"], "z")
    )
    cs.var_name = "x"
    assert all(np.asarray(cs.var_name) == np.asarray(["x0", "x1", "x2"]))

    cs.var_name = ["y0", "y1", "y2"]
    assert all(np.asarray(cs.var_name) == np.asarray(["y0", "y1", "y2"]))

    with pytest.raises(AssertionError):
        cs.var_name = ["y0", "y1"]


def test_eq_ne():
    cs = RealSpace([[1e-10, 1e-1]] * 2, "x", 0.01, scale="log")
    cs2 = RealSpace([1e-10, 1e-1], "x", 0.01, scale="log") * 2
    assert cs.data[0] == cs2.data[0]
    assert cs.data[0] in cs2.data
    assert cs == cs2


def test_contains():
    cs = (
        IntegerSpace([-10, 10], "y")
        + DiscreteSpace(["A", "B", "C", "D", "E"], "z")
        + RealSpace([1e-10, 1e-1], "x", 0.01, scale="log")
    )
    assert RealSpace([1e-10, 1e-1], "x", 0.01, scale="log") in cs


def test_in():
    cs = (
        IntegerSpace([-10, 10], "y")
        + DiscreteSpace(["A", "B", "C", "D", "E"], "z")
        + RealSpace([1e-10, 1e-1], "x", 0.01, scale="log")
    )
    x = Solution(cs.sample(1)[0], var_name=cs.var_name)
    assert RealSpace([1e-10, 1e-1], "x", 0.01, scale="log") in cs
    assert "x" in cs
    assert "xx" not in cs
    assert x.tolist() in cs
    assert x.to_dict() in cs


def test_sample_with_constraints():
    g = lambda x: x - 0.1
    cs = RealSpace([1e-10, 1e-1], "x", 0.01, scale="log")
    X = cs.sample(10, g=g)
    assert all(list(map(lambda x: g(x) <= 0, X)))


def test_SearchSpace_remove():
    cs = (
        RealSpace([1e-10, 1e-1], "x", 1e-3, scale="log")
        + IntegerSpace([-10, 10], "y")
        + DiscreteSpace(["A", "B", "C", "D", "E"], "z")
    )

    cs.remove(0)
    assert isinstance(cs[[0]], IntegerSpace)

    cs.remove("z")
    assert isinstance(cs, IntegerSpace)

    with pytest.raises(KeyError):
        cs.remove("aaa")


def test_SearchSpace_mul():
    cs = (
        RealSpace([0, 5], "x") + IntegerSpace([-10, 10], "y") + DiscreteSpace(["A", "B", "C"], "z")
    )
    __ = ["x0", "y0", "z0", "x1", "y1", "z1"]
    assert (cs * 2).dim == 6
    assert all(np.array((2 * cs).var_name) == np.asarray(__))

    cs *= 2
    assert cs.dim == 6


def test_SearchSpace_sub():
    for _ in range(3):
        cs = (
            RealSpace([0, 5], "x")
            + IntegerSpace([-10, 10], "y")
            + DiscreteSpace(["A", "B", "C"], "z")
        )
        _cs = DiscreteSpace(["A", "B", "C"], "z") + IntegerSpace([10, 20], "p")

        cs2 = cs - cs[1:]
        assert cs2.var_name[0] == "x"
        assert cs.dim == 3

        assert set((cs - _cs).var_name) == set(["x", "y"])
        cs -= _cs
        assert set(cs.var_name) == set(["x", "y"])

        cs -= cs[1:]
        assert cs.var_name[0] == "x"


def test_SearchSpace_concat():
    cs_list = [RealSpace([0, 5], "x") for _ in range(3)]
    cs = SearchSpace.concat(*cs_list)
    assert all(np.array(cs.var_name) == np.array(["x0", "x1", "x2"]))
    assert isinstance(cs, RealSpace)

    cs_list = [RealSpace([0, 5], "x"), DiscreteSpace([1, 2, 3]), IntegerSpace([-5, 5])]
    cs = SearchSpace.concat(*cs_list)
    assert isinstance(cs, SearchSpace)


def test_SearchSpace_iadd():
    cs = RealSpace([0, 5], "x")
    cs += RealSpace([5, 10])
    assert isinstance(cs, RealSpace)
    cs += IntegerSpace([-5, 5])
    assert isinstance(cs, SearchSpace)


def test_SearchSpace_slice():
    cs = (
        RealSpace([1, 5], "x", 2, scale="log")
        + IntegerSpace([-10, 10], "y")
        + DiscreteSpace(["A", "B", "C"], "z")
    )
    assert isinstance(cs[[0]], RealSpace)
    assert isinstance(cs[[1]], IntegerSpace)
    assert isinstance(cs[[2]], DiscreteSpace)

    assert isinstance(cs[0], Real)
    assert isinstance(cs[1], Integer)
    assert isinstance(cs[2], Discrete)
    assert isinstance(cs["z"], Discrete)

    cs = (
        RealSpace([1, 5], "x", 2, scale="log") * 2
        + IntegerSpace([-10, 10], "y")
        + DiscreteSpace(["A", "B", "C"], "z")
    )
    assert isinstance(cs[:2], RealSpace)
    assert isinstance(cs[["x0", "x1"]], RealSpace)
    assert isinstance(cs[[False, False, True, False]], IntegerSpace)

    assert isinstance(cs.filter(["x0", "x1"]), RealSpace)


def test_sample():
    cs = (
        RealSpace([1e-10, 1e-1], "x", 1e-3, scale="log")
        + IntegerSpace([-10, 10], "y")
        + DiscreteSpace(["A", "B", "C", "D", "E"], "z")
    )
    X = cs.sample(10)
    assert np.asarray(X).shape == (10, 3)
    cs.sample(5, method="LHS")
    X = cs.sample(5, method="uniform")
    cs.to_linear_scale(X)


def test_constraints():
    cs = RealSpace([-5, 5], "x") * 2
    g = lambda x: x[0] + x[1] - 5
    X = cs.sample(10, g=g)
    assert all([g(x) <= 0 for x in X])

    X = cs.sample(10, g=lambda x: x[0] + 5.1)
    assert len(X) == 0


def test_scale():
    cs = RealSpace([1e-10, 1e-1], "x", scale="log", random_seed=42)
    x = cs.sample(1)
    assert np.isclose(x, 2.3488813e-07)
    assert np.isclose(cs.to_linear_scale(-15.812834391811666), 1.35697948e-07)
    assert np.isclose(cs.to_linear_scale([-15.812834391811666]), 1.35697948e-07)
    assert np.isclose(cs.to_linear_scale((-15.812834391811666)), 1.35697948e-07)

    C = RealSpace([1, 5], scale="log")
    assert getattr(C.data[0], "_bounds_transformed")[0] == 0

    C = RealSpace([0.5, 0.8], scale="logit")
    assert getattr(C.data[0], "_bounds_transformed")[0] == 0

    C = RealSpace([-1, 1], scale="bilog")
    assert getattr(C.data[0], "_bounds_transformed")[0] == -np.log(2)
    assert getattr(C.data[0], "_bounds_transformed")[1] == np.log(2)

    C = RealSpace([-1, 1], scale="bilog") * 2
    x = C.to_linear_scale([-np.log(2), np.log(2)])
    assert np.all(x == np.array([-1, 1]))


def test_precision():
    cs = RealSpace([0, 1], precision=2) * 3
    X = cs.sample(1, method="LHS")
    X = [re.sub(r"^-?\d+\.(\d+)$", r"\1", str(_)) for _ in X[0]]
    assert all([len(x) <= 2 for x in X])

    X = cs.round(np.random.randn(3))
    X = [re.sub(r"^-?\d+\.(\d+)$", r"\1", str(_)) for _ in X[0]]
    assert all([len(x) <= 2 for x in X])

    X = np.random.rand(2, 3)
    assert isinstance(cs.round(X), np.ndarray)

    X = Solution(cs.sample(10, method="LHS"))
    cs.round(X)

    cs = (
        RealSpace([0, 1], "x", precision=2)
        + IntegerSpace([-10, 10], "y")
        + DiscreteSpace(["A", "B", "C", "D", "E"], "z")
    )

    X = cs.sample(1, method="LHS")[0][0]
    X = re.sub(r"^-?\d+\.(\d+)$", r"\1", str(X))
    assert len(X) <= 2


def test_iter():
    cs = (
        RealSpace([1e-10, 1e-1], "x", 1e-3, scale="log")
        + IntegerSpace([-10, 10], "y")
        + DiscreteSpace(["A", "B", "C", "D", "E"], "z")
    )
    cs *= 2

    for var in iter(cs):
        assert isinstance(var, Variable)


def test_from_dict():
    cs = SearchSpace.from_dict(
        {
            "activation": {
                "type": "c",
                "range": [
                    "elu",
                    "selu",
                    "softplus",
                    "softsign",
                    "relu",
                    "tanh",
                    "sigmoid",
                    "hard_sigmoid",
                    "linear",
                ],
                "N": 3,
            }
        }
    )

    assert cs.dim == 3
    assert cs.var_name[0] == "activation0"

    with open("./shiny/example.json") as f:
        data = json.load(f)

    cs = SearchSpace.from_dict(data["search_param"])


def test_update():
    cs = RealSpace([0, 5], "x") * 3
    cs2 = RealSpace([-100, 100], "x1") + IntegerSpace([0, 10], "y")
    cs.update(cs2)
    assert "y" in cs.var_name
    assert cs[[1]].data[0].bounds[0] == -100
    assert cs[[1]].data[0].bounds[1] == 100


def test_filter():
    cs = (
        RealSpace([1e-10, 1e-1], "x", 0.01, scale="log")
        + IntegerSpace([-10, 10], "y")
        + DiscreteSpace(["A", "B", "C", "D", "E"], "z")
    )
    cs *= 2
    assert cs.filter(["x1"]).var_name == ["x1"]
    assert "x1" not in cs.filter(["x1"], invert=True).var_name


def test_subset():
    x = SubsetSpace(["a", "b", "c", "d"]) * 2
    x.sample(10)
    xx = Subset(["a", "b", "c", "d"])
    xx.sample(10)


def test_node():

    info = {
        "ccc2": [{"name": "cccc", "condition": "ccc2 == 2"}],
        "c1": [
            {"name": "cc1", "condition": "c1 == 1"},
            {"name": "cc2", "condition": "c1 == 10"},
        ],
        "p1": [
            {"name": "c1", "condition": "p1 == 1"},
            {"name": "c2", "condition": "p1 == 2"},
            {"name": "c3", "condition": "p1 == 3"},
        ],
        "cc2": [
            {"name": "ccc1", "condition": "cc2 == 1"},
            {"name": "ccc2", "condition": "cc2 == 2"},
        ],
        "p2": [{"name": "a", "condition": "p2 == 'a'"}, {"name": "b", "condition": "p2 == 'b'"}],
    }
    root = Node.from_dict(info)
    assert root[0].name == "p1"
    assert root[1].name == "p2"

    # print(root[0].get_all_path())


# def test_condition():
#     import string
#     v = Real([0, 5], "x", conditions="y == 'A'")
#     breakpoint()
#     cs = SearchSpace(
#         [Real([0, 5], "x", conditions="`y` == 'A'"), Discrete(list(string.ascii_uppercase), "y")]
#     )
