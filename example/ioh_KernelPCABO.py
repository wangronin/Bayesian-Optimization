import sys
from ioh import Experiment, logger, get_problem
import numpy as np
import copy

sys.path.insert(0, "./")

from bayes_optim.extension import RealSpace, KernelPCABO1, KernelPCABO, KernelFitStrategy, PCABO, BO
from bayes_optim.surrogate import GaussianProcess, trend

import benchmark.bbobbenchmarks as bn
import random
from functools import partial

optimizer_name = sys.argv[1]
seed = 0
random.seed(seed)
np.random.seed(seed)
lb, ub = -5, 5
# OBJECTIVE_FUNCTION = bn.F21()


def create_algorithm(func, dim, total_budget, doe_size):
    space = RealSpace([lb, ub], random_seed=seed) * dim
    if optimizer_name == 'KernelPCABOCheat':
        return KernelPCABO1(
                search_space=space,
                obj_fun=func,
                DoE_size=doe_size,
                max_FEs=total_budget,
                verbose=False,
                n_point=1,
                acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
                max_information_loss = 0.1,
                kernel_fit_strategy = KernelFitStrategy.AUTO,
                NN=dim
            )
    elif optimizer_name == 'KernelPCABOInverse':
        return KernelPCABO(
                search_space=space,
                obj_fun=func,
                DoE_size=doe_size,
                max_FEs=total_budget,
                verbose=False,
                n_point=1,
                acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
                max_information_loss = 0.1,
                kernel_fit_strategy = KernelFitStrategy.AUTO,
                NN=dim
            )
    elif optimizer_name == 'LinearPCABO':
        return PCABO(
                search_space=space,
                obj_fun=func,
                DoE_size=doe_size,
                max_FEs=total_budget,
                verbose=False,
                n_point=1,
                n_components=0.9,
                acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
            )
    elif optimizer_name == 'BO':
        bounds = np.asarray([(lb, ub)]*dim)
        return BO(
                search_space=space,
                obj_fun=func,
                DoE_size=doe_size,
                max_FEs=total_budget,
                verbose=False,
                n_point=1,
                model=GaussianProcess(
                        mean=trend.constant_trend(dim),
                        corr="matern",
                        thetaL=1e-3 * (bounds[:, 1] - bounds[:, 0]),
                        thetaU=1e3 * (bounds[:, 1] - bounds[:, 0]),
                        nugget=1e-6,
                        noise_estim=False,
                        optimizer="BFGS",
                        wait_iter=3,
                        random_start=max(10, dim),
                        likelihood="concentrated",
                        eval_budget=100 * dim,
                    ),
                acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
            )
    else:
        raise NotImplementedError


def fitness_function_wrapper(x, f):
    if type(x) is np.ndarray:
        return f(x.tolist())
    return f(x)


class AlgorithmWrapper:
    def __init__(self):
        self.opt = None

    def __call__(self, f):
        self.dim = f.meta_data.n_variables
        func = partial(fitness_function_wrapper, f=f)
        total_budget = 50 + 10 * self.dim
        doe_size = int(total_budget * 0.2)
        self.opt = create_algorithm(func, self.dim, total_budget, doe_size)
        self.opt.run()

    @property
    def lower_space_dim(self) -> int:
        if self.opt is None:
            return -1
        if optimizer_name == 'BO':
            return self.dim
        return self.opt.get_lower_space_dimensionality()

    @property
    def extracted_information(self) -> float:
        if self.opt is None:
            return -1
        if optimizer_name == 'BO':
            return 1.0
        return self.opt.get_extracted_information()


exp = Experiment(
    algorithm = AlgorithmWrapper(),
    fids = [21],
    iids = [1],
    dims = [10],
    reps = 5,
    problem_type = 'BBOB',
    njobs = 1,
    logged = True,
    folder_name = 'IOH_data_night_experiment',
    algorithm_name = optimizer_name,
    store_positions = True,
    logged_attributes = ['lower_space_dim', 'extracted_information'],
    merge_output = True,
    zip_output = True,
    remove_data = False,
    logger_triggers=[logger.trigger.ON_IMPROVEMENT]
)

exp.run()

