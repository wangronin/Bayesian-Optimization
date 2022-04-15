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
from my_logger import MyIOHFormatOnEveryEvaluationLogger, MyObjectiveFunctionWrapper

import random
from functools import partial
import json
from experiment_helpers import run_particular_experiment, validate_optimizers


def run_experiment():
    if len(sys.argv) == 1:
        print('No configs given')
        return
    with open(sys.argv[1], 'r') as f:
        config = json.load(f)
    result_folder_prefix = config['folder']
    fids = config['fids']
    iids = config['iids']
    dims = config['dims']
    reps = config['reps']
    optimizers = config['optimizers']
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
                        run_particular_experiment(my_optimizer_name, fid, iid, dim, rep, result_folder_prefix)
                        end = time.time()
                        print(f'    Done in {end - start} secs')
                        cur_run_number += 1


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('run_experiment()')
    run_experiment()

