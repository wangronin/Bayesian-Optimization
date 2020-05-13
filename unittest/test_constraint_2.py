import numpy as np

from BayesOpt import BO
from BayesOpt.Surrogate import RandomForest
from BayesOpt.SearchSpace import ContinuousSpace, OrdinalSpace, NominalSpace
from BayesOpt.optimizer import mies

np.random.seed(10)

LENGTH = 3
CACHE = {}

def obj_func(x):
    global LENGTH
    global CACHE
    x_i, f_d = np.array(x[:LENGTH]), x[LENGTH:LENGTH*2]
    cnt = 0
    fitness = 0
    _id = ""
    for n, f in zip(x_i, f_d):
        if f == 'Y':
            fitness += np.power(n, cnt)
            cnt += 1
            _id += str(n) if len(_id) == 0 else '-' + str(n)
    print(x, fitness)
    CACHE[_id] = fitness
    return fitness

def eq_func(x):
    global LENGTH
    global CACHE
    x_i, f_d = np.array(x[:LENGTH]), x[LENGTH:LENGTH*2]
    last_y = -1
    penalty = 0
    _id = ""
    for ix, p in enumerate(zip(x_i, f_d)):
       n, f = p
       if f == 'Y':
          penalty += ix - last_y - 1
          last_y = last_y + 1
          _id += str(n) if len(_id) == 0 else '-' + str(n)

    penalty = (len(f_d) * (len(f_d) + 1)) / 2 if last_y == -1 else penalty
    penalty = (len(f_d) * (len(f_d) + 1)) / 2 if _id in CACHE else penalty
    return int(penalty)


space = (OrdinalSpace([1, 3]) * LENGTH) + (NominalSpace(['Y', 'N']) * LENGTH)

model = RandomForest(levels=space.levels)
opt = BO(space, obj_func, model, eq_func=eq_func, ineq_func=None, minimize=True,
         n_init_sample=3, max_eval=50, verbose=True, optimizer='MIES')
xopt, fopt, stop_dict = opt.run()
print(xopt, fopt, stop_dict)
