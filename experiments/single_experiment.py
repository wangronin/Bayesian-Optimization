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
from experiment_helpers import run_particular_experiment


def run_experiment():
    if len(sys.argv) == 1:
        print('No configs given')
        return
    with open(sys.argv[1]) as f:
        m = json.load(f)
    print(f'Running with config {m} ...')
    start = time.time()
    run_particular_experiment(
        m['opt'], m['fid'], m['iid'], m['dim'], m['seed'], m['folder'])
    end = time.time()
    sec = int(round(end - start))
    x = str(timedelta(seconds=sec)).split(':')
    print(
        f'    Done in {sec} seconds. Which is {x[0]} hours, {x[1]} minutes and {x[2]} seconds')


if __name__ == '__main__':
    run_experiment()
