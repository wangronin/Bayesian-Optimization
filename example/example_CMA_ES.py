import sys
import numpy as np

sys.path.insert(0, '../')
from deap import benchmarks
from bayes_optim.acquisition_optim import OnePlusOne_CMA, OnePlusOne_Cholesky_CMA

obj_fun = lambda x: benchmarks.himmelblau(x)[0]

# def obj_fun(x):
#     return np.sum(x ** 2)

opt = OnePlusOne_Cholesky_CMA(2, obj_fun, lb=-6, ub=6, sigma0=0.2, ftarget=1e-8, verbose=True)

opt.run()
print(opt.stop_dict)