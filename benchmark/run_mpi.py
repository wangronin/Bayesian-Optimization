import os
import sys
from time import time

import numpy as np

import bbobbenchmarks as bn
import fgeneric


def run_optimizer(optimizer, dim, fID, instance, logfile, lb, ub, max_FEs, data_path, bbob_opt):
    """Parallel BBOB/COCO experiment wrapper"""
    # Set different seed for different processes
    start = time()
    seed = np.mod(int(start) + os.getpid(), 1000)
    np.random.seed(seed)

    data_path = os.path.join(data_path, str(instance))
    max_FEs = eval(max_FEs)

    f = fgeneric.LoggingFunction(data_path, **bbob_opt)
    f.setfun(*bn.instantiate(fID, iinstance=instance))

    opt = optimizer(dim, f.evalfun, f.ftarget, max_FEs, lb, ub, logfile)
    opt.run()

    f.finalizerun()
    with open(logfile, "a") as fout:
        fout.write(
            "{} on f{} in {}D, instance {}: FEs={}, fbest-ftarget={:.4e}, "
            "elapsed time [m]: {:.3f}\n".format(
                optimizer,
                fID,
                dim,
                instance,
                f.evaluations,
                f.fbest - f.ftarget,
                (time() - start) / 60.0,
            )
        )


def test_BO(dim, obj_fun, ftarget, max_FEs, lb, ub, logfile):
    sys.path.insert(0, "../")
    from bayes_optim import BO, AnnealingBO, RealSpace
    from bayes_optim.Surrogate import GaussianProcess, trend

    space = RealSpace([lb, ub]) * dim

    mean = trend.constant_trend(dim, beta=0)  # equivalent to Ordinary Kriging
    thetaL = 1e-10 * (ub - lb) * np.ones(dim)
    thetaU = 10 * (ub - lb) * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

    model = GaussianProcess(
        mean=mean,
        corr="matern",
        theta0=theta0,
        thetaL=thetaL,
        thetaU=thetaU,
        noise_estim=False,
        nugget=1e-6,
        optimizer="BFGS",
        wait_iter=5,
        random_start=5 * dim,
        likelihood="concentrated",
        eval_budget=100 * dim,
    )

    return BO(
        search_space=space,
        obj_fun=obj_fun,
        model=model,
        DoE_size=dim * 5,
        max_FEs=max_FEs,
        verbose=False,
        n_point=1,
        minimize=True,
        ftarget=ftarget,
        logger=logfile,
    )


if __name__ == "__main__":
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dims = (2, 5)
    fIDs = bn.nfreeIDs  # for all fcts
    instance = range(1, size + 1)

    algorithms = [test_BO]

    opts = {
        "max_FEs": "100  * dim",
        "lb": -5,
        "ub": 5,
        "data_path": "",
    }
    opts["bbob_opt"] = {
        "comments": "max_FEs={0}".format(opts["max_FEs"]),
    }

    for algorithm in algorithms:
        opts["data_path"] = "./bbob_data/{}".format(algorithm.__name__)
        opts["bbob_opt"]["algid"] = algorithm.__name__
        for dim in dims:
            for fID in fIDs:
                run_optimizer(algorithm, dim, fID, instance[rank], logfile="./log", **opts)
