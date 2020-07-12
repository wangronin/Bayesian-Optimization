import numpy as np

from BayesOpt import BO
from BayesOpt.SearchSpace import ContinuousSpace, OrdinalSpace, NominalSpace
from BayesOpt.Surrogate import RandomForest

np.random.seed(666)

if 11 < 2: # test for flat fitness
    def fitness(x):
        return 1

    space = ContinuousSpace([-5, 5]) * 2
    levels = space.levels if hasattr(space, 'levels') else None
    model = RandomForest(levels=levels)

    opt = BO(space, fitness, model, max_eval=300, verbose=True, n_job=1, n_point=1)
    print(opt.run())

if 1 < 2:
    def fitness1(x):
        x_r, x_i, x_d = np.array(x[:2]), x[2], x[3]
        if x_d == 'OK':
            tmp = 0
        else:
            tmp = 1
        return np.sum(x_r ** 2) + abs(x_i - 10) / 123. + tmp * 2

    space = (ContinuousSpace([-5, 5]) * 2) + OrdinalSpace([5, 15]) + \
        NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])

    levels = space.levels if hasattr(space, 'levels') else None
    model = RandomForest(levels=levels)

    opt = BO(space, fitness1, model, max_eval=300, verbose=True, n_job=1, n_point=3,
                n_init_sample=3,
                init_points=[[0, 0, 10, 'OK']])
    xopt, fopt, stop_dict = opt.run()
