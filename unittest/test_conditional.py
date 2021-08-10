import os
import string
import sys

sys.path.insert(0, "../")
import numpy as np
from bayes_optim.extension import ConditionalBO
from bayes_optim.search_space import Discrete, Integer, Real, SearchSpace

np.random.seed(123)


def fitness(_):
    return np.random.rand()


# def test_conditional():
#     cs = SearchSpace(
#         [
#             Integer([1, 3], "x"),
#             Discrete(["A", "B"], "y1", conditions="x == 1"),
#             Discrete(["A", "B", "C"], "y2", conditions="x == 2"),
#             Discrete(["A", "B", "C"], "y3", conditions="x == 3"),
#             Real([-5, 5], "z1", conditions="y1 == 'A'"),
#             Real([-5, 5], "z2", conditions="y1 == 'B'"),
#             Real([-5, 5], "xx"),
#             Real([-5, 5], "yy"),
#             Real([-5, 5], "zz"),
#         ],
#     )
#     subcs = cs.get_unconditional_subspace()
#     bo = ConditionalBO(subcs, obj_fun=fitness, DoE_size=5, max_FEs=10, verbose=True, n_point=3)
#     bo.run()
