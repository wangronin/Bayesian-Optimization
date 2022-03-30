import sys
import os
from ioh import Experiment, get_problem, logger, problem, OptimizationType
from bayes_optim.acquisition import OnePlusOne_Cholesky_CMA
import numpy as np
import copy
import time

sys.path.insert(0, "./")

from bayes_optim.extension import RealSpace, KernelPCABO1, KernelPCABO, KernelFitStrategy, PCABO, BO
from bayes_optim.surrogate import GaussianProcess, trend

import benchmark.bbobbenchmarks as bn
import random
from functools import partial


MY_EXPEREMENT_FOLDER = "lunchtime_experiment_30_04"
fids = [21]
iids = [0]
dims = [10]
reps = 5
problem_type = 'BBOB'
optimizers = sys.argv[1:]
optimizer_name = sys.argv[1]
seed = 0
random.seed(seed)
np.random.seed(seed)
lb, ub = -5, 5


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
    elif optimizer_name == 'CMA_ES':
        return OnePlusOne_Cholesky_CMA(
            search_space=space,
            obj_fun=obj_fun,
            lb=lb,
            ub=ub,
            sigma0=40,
            ftarget=1e-8,
            verbose=False,
        )
    else:
        raise NotImplementedError


def validate_optimizers(optimizers):
    for optimizer in optimizers:
        global optimizer_name
        optimizer_name = optimizer
        create_algorithm(lambda x: 1, 10, 10, 10)


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

    def __call__(self, f, fid, iid, dim):
        self.dim = dim
        func = partial(AlgorithmWrapper.__fitness_function_wrapper, f=f)
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


def get_function_name(fid, iid):
    return 'F' + str(fid) + '_' + str(iid)


class MyFitnessWrapper:
    def __init__(self, fid, iid, dim, directed_by='Hao'):
        self.fid = fid
        self.iid = iid
        self.dim = dim
        self.my_loggers = []
        if directed_by == 'Hao':
            self.my_function, self.optimum = bn.instantiate(ifun=fid, iinstance=iid)
            iohf = get_problem(fid, dimension=dim, instance=iid, problem_type = 'Real')
            self.func_name = iohf.meta_data.name
        elif directed_by == 'IOH':
            _, self.optimum = bn.instantiate(ifun=fid, iinstance=iid)
            self.my_function = get_problem(fid, dimension=dim, instance=iid, problem_type = 'Real')
            self.func_name = self.my_function.name
        else:
            raise ValueError('Unknown way to create function using', directed_by)
        self.cnt_eval = 0
        self.best_so_far = float('inf')

    def __call__(self, x):
        cur_value = self.my_function(x)
        self.best_so_far = min(self.best_so_far, cur_value)
        self.cnt_eval += 1
        for l in self.my_loggers:
            l.log(self.cnt_eval, cur_value, self.best_so_far)
        return cur_value

    def attach_logger(self, logger):
        self.my_loggers.append(logger)
        logger._set_up_logger(self.fid, self.iid, self.dim, self.func_name)


class MyLogger:
    def __init__(self, folder_name='TMP', algorithm_name='UNKNOWN', suite='unkown suite', algorithm_info='algorithm_info'):
        self.folder_name = MyLogger.__generate_dir_name(folder_name)
        self.algorithm_name = algorithm_name
        self.algorithm_info = algorithm_info
        self.suite = suite
        self.create_time = time.time()

    @staticmethod
    def __generate_dir_name(name, x=0):
        while True:
            dir_name = (name + ('-' + str(x))).strip()
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
                return dir_name
            else:
                x = x + 1

    def watch(self, algorithm, extra_data):
        # self.extra_info_getters = [getattr(algorithm, attr) for attr in extra_data]
        self.algorithm = algorithm
        self.extra_info_getters = extra_data

    def _set_up_logger(self, fid, iid, dim, func_name):
        self.log_info_path = f'{self.folder_name}/IOHprofiler_f{fid}_{func_name}.info'
        with open(self.log_info_path, 'a') as f:
            f.write(f'suite = \"{self.suite}\", funcId = {fid}, funcName = \"{func_name}\", DIM = {dim}, maximization = \"F\", algId = \"{self.algorithm_name}\", algInfo = \"{self.algorithm_info}\"\n')
        self.log_file_path = f'data_f{fid}_{func_name}/IOHprofiler_f{fid}_DIM{dim}.dat'
        self.log_file_full_path = f'{self.folder_name}/{self.log_file_path}'
        os.makedirs(os.path.dirname(self.log_file_full_path), exist_ok=True)
        self.first_line = 0
        self.last_line = 0
        with open(self.log_file_full_path, 'a') as f:
            f.write('\"function evaluation\" \"current f(x)\" \"best-so-far f(x)\" \"current af(x)+b\" \"best af(x)+b\" lower_space_dim extracted_information\n')

    def log(self, cur_evaluation, cur_fitness, best_so_far):
        with open(self.log_file_full_path, 'a') as f:
            f.write(f'{cur_evaluation} {cur_fitness} {best_so_far} {cur_fitness} {best_so_far}')
            for fu in self.extra_info_getters:
                try:
                    extra_info = getattr(self.algorithm, fu)
                except Exception as e:
                    extra_info = 'None'
                f.write(f' {extra_info}')
            f.write('\n')
            self.last_line += 1

    def finish_logging(self):
        time_taken = time.time() - self.create_time
        with open(self.log_info_path, 'a') as f:
            f.write('%\n')
            f.write(f'{self.log_file_path}, {self.first_line}:{self.last_line}|{time_taken}')


def run_particular_experiment(my_optimizer_name, fid, iid, dim, rep):
    global seed, optimizer_name
    seed = rep
    optimizer_name = my_optimizer_name
    algorithm = AlgorithmWrapper()
    l = MyLogger(folder_name=MY_EXPEREMENT_FOLDER, algorithm_name=optimizer_name)
    print(f'    Logging to the folder {l.folder_name}')
    l.watch(algorithm, ['lower_space_dim', 'extracted_information'])
    p = MyFitnessWrapper(fid, iid, dim)
    p.attach_logger(l)
    algorithm(p, fid, iid, dim)
    l.finish_logging()



def run_experiment():
    validate_optimizers(optimizers)
    runs_number = len(optimizers) * len(fids) * len(iids) * len(dims) * reps
    cur_run_number = 1
    for my_optimizer_name in optimizers:
        for fid in fids:
            for iid in iids:
                for dim in dims:
                    for rep in range(reps):
                        print(f'Run {cur_run_number} out of {runs_number}, Algorithm {my_optimizer_name}, Problem {fid}, Instance {iid}, Dimension {dim}, Repetition {rep+1} ...')
                        start = time.time()
                        run_particular_experiment(my_optimizer_name, fid, iid, dim, rep)
                        end = time.time()
                        print(f'    Done in {end - start} secs')
                        cur_run_number += 1


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('run_experiment()')
    run_experiment()

