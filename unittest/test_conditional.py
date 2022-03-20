import sys

sys.path.insert(0, "./")
import numpy as np
from bayes_optim.extension import ConditionalBO
from bayes_optim.search_space import Discrete, Integer, Real, SearchSpace

np.random.seed(123)


def fitness(params):
    return (
        params["x"] ** 2
        + (params["y1"] == "B" if params["y1"] else 0)
        + (params["y2"] == "A" if params["y2"] else 0)
        + (params["y3"] == "C" if params["y3"] else 0)
        + (params["z1"] ** 3 if params["z1"] else 0)
        + (params["z2"] * 10 if params["z2"] else 0)
        + params["xx"] * params["yy"] * params["zz"]
    )


def test_conditional():
    space = SearchSpace(
        [
            Integer([1, 3], "x"),
            Discrete(["A", "B", "C"], "y1", conditions="x == 1"),
            Discrete(["A", "B", "C"], "y2", conditions="x == 2"),
            Discrete(["A", "B", "C"], "y3", conditions="x == 3"),
            Real([-5, 5], "z1", conditions="y1 == 'A'"),
            Real([-5, 5], "z2", conditions="y1 == 'B'"),
            Real([-5, 5], "xx"),
            Real([-5, 5], "yy"),
            Real([-5, 5], "zz"),
        ],
    )
    bo = ConditionalBO(search_space=space, obj_fun=fitness, DoE_size=5, max_FEs=20, verbose=True, n_point=3)
    bo.run()
