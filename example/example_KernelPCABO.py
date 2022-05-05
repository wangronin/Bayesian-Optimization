import random
import benchmark.bbobbenchmarks as bn
from bayes_optim.mylogging import eprintf
from bayes_optim.extension import PCABO, RealSpace, KernelPCABO, KernelFitStrategy
import sys

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from bayes_optim import RandomForest, BO, GaussianProcess

sys.path.insert(0, "./")


SEED = int(sys.argv[1])
random.seed(SEED)
np.random.seed(SEED)
dim = 2
lb, ub = -5, 5
OBJECTIVE_FUNCTION = bn.F21()


def func(x):
    #     x = np.asarray(x)
    #     return np.sum((np.arange(1, dim + 1) * x) ** 2)
    return OBJECTIVE_FUNCTION(x)


space = RealSpace([lb, ub], random_seed=SEED) * dim
doe_size = 5
total_budget = 40
dim = 2
eprintf("new call to Kernel PCABO")
opt = KernelPCABO(
    search_space=space,
    obj_fun=func,
    DoE_size=doe_size,
    max_FEs=total_budget,
    verbose=True,
    n_point=1,
    acquisition_optimization={"optimizer": "BFGS"},
    max_information_loss=0.5,
    kernel_fit_strategy=KernelFitStrategy.AUTO,
    NN=dim,
    random_seed=SEED
)

print(opt.run())
