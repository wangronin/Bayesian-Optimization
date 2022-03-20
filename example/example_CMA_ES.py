import sys

import numpy as np

sys.path.insert(0, "./")
from bayes_optim.acquisition import OnePlusOne_Cholesky_CMA
from bayes_optim.search_space import RealSpace


def obj_fun(x):
    return np.sum(x**2)


opt = OnePlusOne_Cholesky_CMA(
    search_space=RealSpace([-5, 5]) * 30,
    obj_fun=obj_fun,
    lb=-100,
    ub=100,
    sigma0=40,
    ftarget=1e-8,
    verbose=True,
)

opt.run()
print(opt.stop_dict)
