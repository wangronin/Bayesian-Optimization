import sys

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from bayes_optim import RandomForest, BO, GaussianProcess

sys.path.insert(0, "./")

from bayes_optim.extension import PCABO, RealSpace, KernelPCABO1
from bayes_optim.mylogging import eprintf
from bayes_optim.extension import KernelFitStrategy

import benchmark.bbobbenchmarks as bn
import random


SEED = int(sys.argv[1])
random.seed(SEED)
np.random.seed(SEED)
dim = 2
lb, ub = -5, 5
OBJECTIVE_FUNCTION = bn.F21()


def fitness(x):
#     x = np.asarray(x)
#     return np.sum((np.arange(1, dim + 1) * x) ** 2)
    return OBJECTIVE_FUNCTION(x)


space = RealSpace([lb, ub], random_seed=SEED) * dim
eprintf("new call to Kernel PCABO")
doe_size = 5
total_budget = 40
opt = KernelPCABO1(
    search_space=space,
    obj_fun=OBJECTIVE_FUNCTION,
    DoE_size=doe_size,
    max_FEs=total_budget,
    verbose=True,
    n_point=1,
    acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
    max_information_loss = 0.9,
    kernel_fit_strategy=KernelFitStrategy.AUTO,
    # kernel_config={'kernel_name': 'rbf', 'kernel_parameters': {'gamma': 0.05}},
    NN=dim
)


print(opt.run())

