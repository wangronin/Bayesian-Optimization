import sys

import numpy as np

sys.path.insert(0, "./")

from bayes_optim import RealSpace
from bayes_optim.mobo import MOBO
from bayes_optim.surrogate import RandomForest

np.random.seed(123)
dim = 2
lb, ub = -10, 10


def f1(x):
    x = np.asarray(x)
    return np.sum((x + 5) ** 2)


def f2(x):
    x = np.asarray(x)
    return np.sum((x - 5) ** 2)


space = RealSpace([lb, ub]) * dim
# Bayesian optimization also uses a Surrogate model
# For mixed variable type, the random forest is typically used
model = RandomForest(levels=space.levels)

opt = MOBO(
    search_space=space,
    obj_fun=(f1, f2),
    model=model,
    DoE_size=5,
    max_FEs=100,
    verbose=True,
    acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
)
print(opt.run())
