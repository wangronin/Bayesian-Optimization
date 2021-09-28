import os
import sys

import numpy as np

sys.path.insert(0, "./")

from bayes_optim import BO, RealSpace
from bayes_optim.surrogate import GaussianProcess

np.random.seed(123)
dim = 5
lb, ub = -1, 5


def fitness(x):
    x = np.asarray(x)
    return np.sum(x ** 2)


space = RealSpace([lb, ub]) * dim
model = GaussianProcess(dim=dim, alpha=1e-3, n_restarts_optimizer=dim)

opt = BO(
    search_space=space,
    obj_fun=fitness,
    model=model,
    DoE_size=5,
    max_FEs=50,
    verbose=True,
    n_point=1,
    acquisition_optimization={"optimizer": "BFGS"},
)
print(opt.run())
