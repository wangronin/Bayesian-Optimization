from functools import partial
import random
from maria_laura.wrapper import marialaura as create_marialaura_alg
from my_logger import MyIOHFormatOnEveryEvaluationLogger, MyObjectiveFunctionWrapper
from bayes_optim.surrogate import GaussianProcess, trend
from bayes_optim.extension import RealSpace, KernelPCABO1, KernelPCABO, KernelFitStrategy, PCABO, BO
import sys
import os
import json
from ioh import Experiment, get_problem, logger, problem, OptimizationType
from bayes_optim.acquisition import OnePlusOne_Cholesky_CMA
import numpy as np
import copy
import time
from datetime import timedelta


lb, ub = -5, 5
cnt = -1


def create_algorithm(optimizer_name, func, dim, total_budget, doe_size, seed):
    space = RealSpace([lb, ub], random_seed=seed) * dim
    print(f'seed={seed}')
    global cnt
    cnt += 1
    if optimizer_name == 'KernelPCABOCheat':
        return KernelPCABO1(
            search_space=space,
            obj_fun=func,
            DoE_size=doe_size,
            max_FEs=total_budget,
            verbose=False,
            n_point=1,
            acquisition_optimization={"optimizer": "BFGS"},
            max_information_loss=0.1,
            kernel_fit_strategy=KernelFitStrategy.AUTO,
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
            acquisition_optimization={"optimizer": "BFGS"},
            max_information_loss=0.1,
            kernel_fit_strategy=KernelFitStrategy.AUTO,
            NN=dim,
            random_seed=seed
        )
    elif optimizer_name == 'LinearPCABO':
        return PCABO(
            search_space=space,
            obj_fun=func,
            DoE_size=doe_size,
            max_FEs=total_budget,
            verbose=False,
            n_point=1,
            n_components=0.90,
            acquisition_optimization={"optimizer": "BFGS"},
            random_seed=seed
        )
    elif optimizer_name == 'BO':
        return BO(
            search_space=space,
            obj_fun=func,
            DoE_size=doe_size,
            n_point=1,
            random_seed=seed,
            data_file=f"test-{cnt}.csv",
            acquisition_optimization={"optimizer": "BFGS"},
            max_FEs=total_budget,
            verbose=False,
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
    elif optimizer_name == 'SAASBO':
        return create_saasbo(
            optimizer_name='saasbo',
            func=func,
            ml_dim=dim,
            ml_total_budget=total_budget,
            ml_DoE_size=doe_size,
            random_seed=seed
        )
    elif optimizer_name == 'SKlearnBO':
        return create_marialaura_alg(
            optimizer_name='BO_sklearn',
            func=func,
            ml_dim=dim,
            ml_total_budget=total_budget,
            ml_DoE_size=doe_size,
            random_seed=seed
        )
    else:
        raise NotImplementedError


class AlgorithmWrapper:
    def __init__(self, seed):
        self.opt = None
        self.seed = seed

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
        doe_size = 3 * self.dim
        if self.dim == 10:
            total_budget = 250
        elif self.dim == 20:
            total_budget = 120
        elif self.dim == 40:
            total_budget = 250
        elif self.dim == 60:
            total_budget = 300
        else:
            total_budget = 2 * doe_size
        self.opt = create_algorithm(
            optimizer_name, func, self.dim, total_budget, doe_size, self.seed)
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
    def kernel_config(self) -> str:
        return self.opt._pca.get_kernel_parameters()

    @property
    def out_of_the_box_solutions(self) -> int:
        return self.opt.out_solutions

    @property
    def acq_opt_time(self) -> float:
        return self.opt.acq_opt_time

    @property
    def model_fit_time(self) -> float:
        return self.opt.mode_fit_time


def run_particular_experiment(my_optimizer_name, fid, iid, dim, rep, folder_name):
    algorithm = AlgorithmWrapper(rep)
    l = MyIOHFormatOnEveryEvaluationLogger(
        folder_name=folder_name, algorithm_name=my_optimizer_name)
    print(f'    Logging to the folder {l.folder_name}')
    sys.stdout.flush()
    l.watch(algorithm, ['lower_space_dim', 'extracted_information',
            'out_of_the_box_solutions', 'kernel_config', 'acq_opt_time', 'model_fit_time'])
    p = MyObjectiveFunctionWrapper(fid, iid, dim)
    p.attach_logger(l)
    algorithm(my_optimizer_name, p, fid, iid, dim)
    l.finish_logging()


def validate_optimizers(optimizers):
    for optimizer in optimizers:
        create_algorithm(optimizer, lambda x: 1, 10, 10, 10, 0)
