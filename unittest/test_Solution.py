import os
import sys

sys.path.insert(0, "../")

import dill
import numpy as np
from bayes_optim import Solution


def test_1D():
    s = Solution(np.random.randint(0, 100, size=(5,)), index=1)
    assert isinstance(s[0], int)
    s.fitness = 3
    assert s.ndim == 1
    assert s.fitness == 3
    assert all(s.unique() == s)


def test_2D():
    # test for 2D solution
    A, B = np.random.randn(5, 3).tolist(), ["simida", "niubia", "bang", "GG", "blyat"]
    s = Solution([A[i] + [B[i]] for i in range(5)], verbose=True, fitness=0, fitness_name="f")
    assert s.ndim == 2
    assert s.N == 5
    assert s.dim == 4

    s[:, 0] = np.asarray(["wa"] * 5).reshape(-1, 1)
    assert np.all(s[:, 0] == "wa")

    s[3].fitness = 12
    assert s.fitness[3] == 12

    a = s[0]
    a.fitness = 3
    assert s.fitness[0] == 3

    s[2:4].fitness = 1
    assert np.all(s.fitness[2:4] == 1)

    s[0].index = "5"
    assert s.index[0] == "5"

    print(s[0:1])
    print(s[0, 0:3])
    print(s[:, 0])
    print(s[0:2][0, 0:2])
    print(s[0][0:2])

    print(s + s[3:5])
    print(s[0:2] + s[3:5])


def test_to_dict():
    s = Solution(np.random.randn(10, 5))
    print(s.to_dict(orient="index"))

    s = Solution(np.random.randn(10))
    s.to_dict()


def test_pickling():
    s = Solution(np.random.randn(10, 5))
    a = dill.dumps(s)
    s2 = dill.loads(a)
    assert np.all(s == s2)


def test_to_csv():
    # # test saving to csv
    s = Solution(np.random.randn(10, 5))
    s.to_csv("test.csv", header=True, attribute=True, index=True)
    os.remove("test.csv")


def test_from_dict():
    s = Solution(np.random.randn(10, 5))
    s_ = s.to_dict(orient="index")
    s2 = Solution.from_dict(s_)
    assert np.all(s == s2)
    assert isinstance(s.to_dict(orient="index", with_index=True), dict)
    assert isinstance(s.to_dict(orient="var", with_index=True), dict)
