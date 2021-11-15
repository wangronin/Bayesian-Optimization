import sys

import numpy as np

sys.path.insert(0, "./")

from bayes_optim.extension import PCABO, RealSpace

np.random.seed(123)
dim = 5
lb, ub = -5, 5


def fitness(x):
    x = np.asarray(x)
    return np.sum((np.arange(1, dim + 1) * x) ** 2)


space = RealSpace([lb, ub]) * dim
opt = PCABO(
    search_space=space,
    obj_fun=fitness,
    DoE_size=5,
    max_FEs=40,
    verbose=True,
    n_point=1,
    n_components=0.95,
    acquisition_optimization={"optimizer": "BFGS"},
)
print(opt.run())
