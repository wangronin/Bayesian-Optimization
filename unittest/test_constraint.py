import numpy as np
import sys, os
sys.path.insert(0, '../')

from bayes_optim import BO, ContinuousSpace, OrdinalSpace, NominalSpace
from bayes_optim.Surrogate import RandomForest

np.random.seed(42)

def obj_func(x):
    return (x['pc'] - 0.2) ** 2 + x['mu'] + x['lambda'] + np.abs(x['p'] - 0.7)

def g(x):
    return [-x['pc'], x['mu'] - 1.9]

def test_BO_constraints():
    search_space = OrdinalSpace([1, 10], var_name='mu') + \
        OrdinalSpace([1, 10], var_name='lambda') + \
            ContinuousSpace([0, 1], var_name='pc') + \
                ContinuousSpace([0.005, 0.5], var_name='p')

    model = RandomForest(levels=search_space.levels)
    xopt, _, __ = BO(
        search_space=search_space,
        obj_fun=obj_func,
        ineq_fun=g,
        model=model,
        max_FEs=30,
        DoE_size=3,
        eval_type='dict',
        acquisition_fun='MGFI',
        acquisition_par={'t' : 2},
        n_job=1,
        n_point=1,
        verbose=True
    ).run()

    assert isinstance(xopt, dict)
    assert all(np.array(g(xopt)) <= 0)

# LENGTH = 3
# CACHE = {}

# def obj_func(x):
#     global LENGTH
#     global CACHE
#     x_i, f_d = np.array(x[:LENGTH]), x[LENGTH:LENGTH*2]
#     cnt = 0
#     fitness = 0
#     _id = ""
#     for n, f in zip(x_i, f_d):
#         if f == 'Y':
#             fitness += np.power(n, cnt)
#             cnt += 1
#             _id += str(n) if len(_id) == 0 else '-' + str(n)
#     print(x, fitness)
#     CACHE[_id] = fitness
#     return fitness

# def eq_func(x):
#     global LENGTH
#     global CACHE
#     x_i, f_d = np.array(x[:LENGTH]), x[LENGTH:LENGTH*2]
#     last_y = -1
#     penalty = 0
#     _id = ""
#     for ix, p in enumerate(zip(x_i, f_d)):
#        n, f = p
#        if f == 'Y':
#           penalty += ix - last_y - 1
#           last_y = last_y + 1
#           _id += str(n) if len(_id) == 0 else '-' + str(n)

#     penalty = (len(f_d) * (len(f_d) + 1)) / 2 if last_y == -1 else penalty
#     penalty = (len(f_d) * (len(f_d) + 1)) / 2 if _id in CACHE else penalty
#     return int(penalty)


# space = (OrdinalSpace([1, 3]) * LENGTH) + (NominalSpace(['Y', 'N']) * LENGTH)

# model = RandomForest(levels=space.levels)
# opt = BO(space, obj_func, model, eq_fun=eq_func, ineq_fun=None, minimize=True,
#          n_init_sample=3, max_eval=50, verbose=True, optimizer='MIES')
# xopt, fopt, stop_dict = opt.run()
# print(xopt, fopt, stop_dict)
