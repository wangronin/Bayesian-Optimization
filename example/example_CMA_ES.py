import sys

import numpy as np

sys.path.insert(0, "./")
from bayes_optim.acquisition_optim import OnePlusOne_Cholesky_CMA


def obj_fun(x):
    return np.sum(x ** 2)


opt = OnePlusOne_Cholesky_CMA(30, obj_fun, lb=-100, ub=100, sigma0=40, ftarget=1e-8, verbose=True)

opt.run()
print(opt.stop_dict)
