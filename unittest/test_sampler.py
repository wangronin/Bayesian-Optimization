import sys

import numpy as np

sys.path.insert(0, "../")

from bayes_optim import DiscreteSpace, IntegerSpace, RealSpace
from bayes_optim.search_space.samplers import SCMC

search_space = (
    RealSpace([-5, 5]) * 2 + DiscreteSpace(["A", "B", "C", "D"]) + IntegerSpace([1, 10]) * 2
)


def h(x):
    return np.array([bool(x[2] not in ["A", "B"]), x[4] ** 2 + x[4] - 2])


def g(x):
    return np.array(
        [
            np.sum(x[:2] ** 2) - 1,
            0.25 - np.sum(x[:2] ** 2),
            x[3] - 5.1,
        ]
    )


def test_SCMC():
    sampler = SCMC(
        search_space,
        lambda x: np.r_[np.abs(h(x)) if h else [], np.array(g(x)) if g else []],
        tol=1e-1,
    )
    X = sampler.sample(10)
    assert all([np.all(np.isclose(h(x), 0, atol=1e-1)) for x in X])
    assert all([np.all(g(x) <= 0) for x in X])


def test_search_space_sampling():
    X = search_space.sample(10, h=h, g=g)
    assert all([np.all(np.isclose(h(x), 0, atol=1e-2)) for x in X])
    assert all([np.all(g(x) <= 0) for x in X])
