import sys

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from bayes_optim import RandomForest, BO, GaussianProcess

sys.path.insert(0, "./")

from bayes_optim.extension import PCABO, RealSpace, KernelPCABO
from bayes_optim.mylogging import eprintf

import benchmark.bbobbenchmarks as bn
import random


SEED = int(sys.argv[1])
random.seed(SEED)
np.random.seed(SEED)
dim = 2
lb, ub = -5, 5
OBJECTIVE_FUNCTION = bn.F17()


def fitness(x):
#     x = np.asarray(x)
#     return np.sum((np.arange(1, dim + 1) * x) ** 2)
    return OBJECTIVE_FUNCTION(x) 


space = RealSpace([lb, ub], random_seed=SEED) * dim
eprintf("new call to Kernel PCABO")
opt = KernelPCABO(
    search_space=space,
    obj_fun=fitness,
    DoE_size=5,
    max_FEs=40,
    verbose=True,
    n_point=1,
    gamma = 0.0001,
    acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
)


print(opt.run())

