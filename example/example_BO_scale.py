import sys

import numpy as np

sys.path.insert(0, "./")

from bayes_optim import BO, Real, SearchSpace
from bayes_optim.surrogate import RandomForest

np.random.seed(123)


def fitness(x):
    x = np.asarray(x)
    return np.sum(x ** 2)


space = SearchSpace(
    [
        Real([0, 10], scale="log", precision=2),
        Real([0, 10], scale="log10", precision=2),
        Real([0, 1], scale="logit", precision=2),
        Real([-10, 10], scale="bilog", precision=2),
    ]
)
space.sample(1)

model = RandomForest(levels=space.levels)
opt = BO(
    search_space=space,
    obj_fun=fitness,
    model=model,
    DoE_size=5,
    max_FEs=50,
    verbose=True,
    n_point=1,
    acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
)
print(opt.run())
