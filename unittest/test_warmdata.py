import numpy as np
import sys, os
sys.path.insert(0, '../')

from bayes_optim import ParallelBO, BO, ContinuousSpace, OrdinalSpace, NominalSpace
from bayes_optim.Surrogate import RandomForest

np.random.seed(42)

def obj_fun(x):
    x_r, x_i, x_d = np.array(x[:2]), x[2], x[3]
    if x_d == 'OK':
        tmp = 0
    else:
        tmp = 1
    return np.sum((x_r + np.array([2, 2])) ** 2) + abs(x_i - 10) * 10 + tmp

def test_warmdata():
    space = ContinuousSpace([-10, 10]) * 2 + \
        OrdinalSpace([5, 15]) + \
        NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])

    X = space.sampling(10)
    y = [obj_fun(x) for x in X]

    model = RandomForest(levels=space.levels)
    opt = BO(
        search_space=space,
        obj_fun=obj_fun,
        model=model,
        minimize=True,
        eval_type='list',
        max_FEs=10,
        verbose=True,
        acquisition_fun='EI',
        warm_data=(X, y)
    )
    opt.run()
    assert opt.data.shape[0] == 20