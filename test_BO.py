import numpy as np
import sys, os

from BayesOpt import AnnealingBO, BO, ContinuousSpace, OrdinalSpace, NominalSpace, RandomForest

np.random.seed(666)
def test__flat_fitness():
    def fitness(x):
        return 1

    space = ContinuousSpace([-5, 5]) * 2
    levels = space.levels if hasattr(space, 'levels') else None
    model = RandomForest(levels=levels)

    opt = BO(
        search_space=space, 
        obj_fun=fitness, 
        model=model, 
        max_FEs=300, verbose=True, 
        n_job=1, 
        n_point=1
    )
    print(opt.run())

def test__mixed_integer():
    def fitness(x):
        x_r, x_i, x_d = np.array(x[:2]), x[2], x[3]
        if x_d == 'OK':
            tmp = 0
        else:
            tmp = 1
        return np.sum(x_r ** 2) + abs(x_i - 10) / 123. + tmp * 2

    space = (ContinuousSpace([-5, 5]) * 2) + \
        OrdinalSpace([5, 15]) + \
        NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])

    levels = space.levels if hasattr(space, 'levels') else None
    model = RandomForest(levels=levels)

    opt = AnnealingBO(
        search_space=space, 
        obj_fun=fitness, 
        model=model, 
        max_FEs=300, 
        verbose=True, 
        n_job=3, 
        n_point=3,
        acquisition_fun='MGFI',
        acquisition_par={'t' : 2},
        DoE_size=3
    )
    opt.run()

test__mixed_integer()