import sys

sys.path.insert(0, "./")

import random

import benchmark.bbobbenchmarks as bn
import numpy as np
from bayes_optim import BO, GaussianProcess, RandomForest
from bayes_optim.extension import PCABO, KernelPCABO, RealSpace
from bayes_optim.mylogging import eprintf
from sklearn.gaussian_process import GaussianProcessRegressor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
dim = 20
lb, ub = -5, 5
OBJECTIVE_FUNCTION = bn.F21()


def fitness(x):
    # x = np.asarray(x)
    # return np.sum((np.arange(1, dim + 1) * x) ** 2)
    # eprintf("Evaluated solution:", x, "type", type(x))
    if type(x) is np.ndarray:
        x = x.tolist()
    return OBJECTIVE_FUNCTION(np.array(x)) - OBJECTIVE_FUNCTION.fopt


space = RealSpace([lb, ub], random_seed=SEED) * dim
eprintf("new call to PCABO")
opt = KernelPCABO(
    # opt = PCABO(
    search_space=space,
    obj_fun=fitness,
    DoE_size=5,
    max_FEs=100,
    verbose=True,
    n_point=1,
    # n_components=1,
    random_seed=42,
    acquisition_optimization={"optimizer": "BFGS"},
)

print(opt.run())
