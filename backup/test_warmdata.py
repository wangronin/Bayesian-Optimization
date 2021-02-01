import numpy as np
from deap import benchmarks

from BayesOpt import BO
from BayesOpt.Surrogate import RandomForest
from BayesOpt.SearchSpace import ContinuousSpace, OrdinalSpace, NominalSpace
from BayesOpt.base import Solution


np.random.seed(42)

def obj_func(x):
    x_r, x_i, x_d = np.array(x[:2]), x[2], x[3]
    if x_d == 'OK':
        tmp = 0
    else:
        tmp = 1
    return np.sum((x_r + np.array([2, 2])) ** 2) + abs(x_i - 10) * 10 + tmp 

def eq_func(x):
    x_r = np.array(x[:2])
    return np.sum(x_r ** 2) - 2

def ineq_func(x):
    x_r = np.array(x[:2])
    return np.sum(x_r) + 1

space = ((ContinuousSpace([-10, 10]) * 2) + OrdinalSpace([5, 15])
    + NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E', 'F', 'G']))


warm_data = Solution([4.6827082694127835, 9.87885354178838, 5, 'A'], var_name=["r_0", "r_1", "i", "d"], n_eval=1, fitness=236.76575128)
warm_data += Solution([-8.99187067168115, 8.317469942991558, 5, 'D'], var_name=["r_0", "r_1", "i", "d"], n_eval=1, fitness=206.33644151)
warm_data += Solution([-2.50919762305275, 9.014286128198322, 12, 'G'], var_name=["r_0", "r_1", "i", "d"], n_eval=1, fitness=142.57378113)
warm_data += Solution([4.639878836228101, 1.973169683940732, 9, 'G'], var_name=["r_0", "r_1", "i", "d"], n_eval=1, fitness=70.8740683)


if 11 < 2:
    model = RandomForest(levels=space.levels)
    opt = BO(space, obj_func, model, minimize=True,
             n_init_sample=3, max_eval=50, verbose=True, optimizer='MIES',
             warm_data=warm_data)
    xopt, fopt, stop_dict = opt.run()
else:
    model = RandomForest(levels=space.levels)
    opt = BO(space, obj_func, model, minimize=True,
             n_init_sample=3, max_eval=50, verbose=True, optimizer='MIES',
             warm_data="test_warmdata.data")
    xopt, fopt, stop_dict = opt.run()
