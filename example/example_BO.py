import sys

sys.path.insert(0, "./")

import random

import benchmark.bbobbenchmarks as bn
import numpy as np
from bayes_optim.extension import KernelPCABO, RealSpace
from bayes_optim.mylogging import eprintf

dim = 10
lb, ub = -5, 5
OBJECTIVE_FUNCTION = bn.F21()


def fitness(x):
    if type(x) is np.ndarray:
        x = x.tolist()
    return OBJECTIVE_FUNCTION(np.array(x)) - OBJECTIVE_FUNCTION.fopt


res = []
for i in range(10):
    space = RealSpace([lb, ub], random_seed=i) * dim
    eprintf("new call to PCABO")
    opt = KernelPCABO(
        search_space=space,
        obj_fun=fitness,
        DoE_size=3 * dim,
        max_FEs=100,
        verbose=True,
        n_point=1,
        random_seed=i,
        data_file=f"test{i}.csv",
        acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
    )
    opt.run()
    res += [opt.xopt.fitness]

breakpoint()
