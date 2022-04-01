import sys
import os
import json
from ioh import Experiment, get_problem, logger, problem, OptimizationType
from bayes_optim.acquisition import OnePlusOne_Cholesky_CMA
import numpy as np
import copy
import time
from datetime import timedelta

sys.path.insert(0, "./")

from bayes_optim.extension import RealSpace, KernelPCABO1, KernelPCABO, KernelFitStrategy, PCABO, BO
from bayes_optim.surrogate import GaussianProcess, trend
from my_logger import MyIOHFormatOnEveryEvaluationLogger, MyObjectiveFunctionWrapper

import random
from functools import partial


MY_EXPEREMENT_FOLDER = "TMP"
seed = 0
lb, ub = -5, 5


def create_algorithm(optimizer_name, func, dim, total_budget, doe_size):
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
                random_seed=seed,
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
    elif optimizer_name == 'CMA_ES':
        return OnePlusOne_Cholesky_CMA(
            search_space=space,
            obj_fun=func,
            lb=lb,
            ub=ub,
            sigma0=40,
            max_FEs=total_budget,
            verbose=False,
            random_seed=seed
        )
    else:
        raise NotImplementedError


def validate_optimizers(optimizers):
    for optimizer in optimizers:
        create_algorithm(optimizer, lambda x: 1, 10, 10, 10)


class AlgorithmWrapper:
    def __init__(self):
        self.opt = None

    @staticmethod
    def __fitness_function_wrapper(x, f):
        if type(x) is np.ndarray:
            x = x.tolist()
        return f(x)

    @staticmethod
    def create_fitness(my_function):
        return partial(AlgorithmWrapper.__fitness_function_wrapper, f=my_function)

    def __call__(self, optimizer_name, f, fid, iid, dim):
        self.dim = dim
        self.optimizer_name = optimizer_name
        func = partial(AlgorithmWrapper.__fitness_function_wrapper, f=f)
        total_budget = 50 + 10 * self.dim
        doe_size = 3 * self.dim
        self.opt = create_algorithm(optimizer_name, func, self.dim, total_budget, doe_size)
        self.opt.run()

    @property
    def lower_space_dim(self) -> int:
        if self.optimizer_name == 'BO':
            return self.dim
        return self.opt.get_lower_space_dimensionality()

    @property
    def extracted_information(self) -> float:
        if self.optimizer_name == 'BO':
            return 1.0
        return self.opt.get_extracted_information()

    @property
    def kernel_config(self) -> float:
        return self.opt._pca.get_kernel_parameters()


def run_particular_experiment(my_optimizer_name, fid, iid, dim, rep):
    global seed
    seed = rep
    algorithm = AlgorithmWrapper()
    l = MyIOHFormatOnEveryEvaluationLogger(folder_name=MY_EXPEREMENT_FOLDER, algorithm_name=my_optimizer_name)
    print(f'    Logging to the folder {l.folder_name}')
    l.watch(algorithm, ['lower_space_dim', 'extracted_information', 'kernel_config'])
    p = MyObjectiveFunctionWrapper(fid, iid, dim)
    p.attach_logger(l)
    algorithm(my_optimizer_name, p, fid, iid, dim)
    l.finish_logging()

def run_experiment():
    if len(sys.argv) == 1:
        print('No configs given')
        return
    with open(sys.argv[1]) as f:
        m = json.load(f)
    print(f'Running with config {m} ...')
    global MY_EXPEREMENT_FOLDER, lb, ub
    MY_EXPEREMENT_FOLDER = m['folder']
    lb = m['lb']
    ub = m['ub']
    start = time.time()
    run_particular_experiment(m['opt'], m['fid'], m['iid'], m['dim'], m['seed'])
    end = time.time()
    sec = int(round(end - start))
    x = str(timedelta(seconds=sec)).split(':')
    print(f'    Done in {sec} seconds. Which is {x[0]} hours, {x[1]} minutes and {x[2]} seconds')


if __name__ == '__main__':
    run_experiment()

