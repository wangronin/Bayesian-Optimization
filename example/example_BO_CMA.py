import sys

import numpy as np

sys.path.insert(0, "./")

from bayes_optim import BO, RealSpace
from bayes_optim.acquisition_optim import OnePlusOne_Cholesky_CMA
from bayes_optim.surrogate import GaussianProcess, trend
import benchmark.bbobbenchmarks as bn
from bayes_optim.mylogging import eprintf

np.random.seed(0)
dim = 5
lb, ub = -5, 5
OBJECTIVE_FUNCTION = bn.F17()

def fitness(x):
#     x = np.asarray(x)
#     return np.sum((np.arange(1, dim + 1) * x) ** 2)
    eprintf("Evaluated solution:", x, "type", type(x))
    if type(x) is np.ndarray:
        x = x.tolist()
    return OBJECTIVE_FUNCTION(np.array(x)) 


space = RealSpace([lb, ub]) * dim
eprintf("new call to PCABO")
#opt = PCABO(
#    search_space=space,
#    obj_fun=fitness,
#    DoE_size=5,
#    max_FEs=40,
#    verbose=True,
#    n_point=1,
#    acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
#)

model = GaussianProcess(
    domain=space,
    n_obj=1,
    n_restarts_optimizer=dim,
)

bo = BO(
    search_space=space,
    obj_fun=fitness,
    model=model,
    #eval_type="list",
    DoE_size=5,
    n_point=1,
    acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
    verbose=True,
    minimize=True,
    max_FEs=40
)

bo.run()

