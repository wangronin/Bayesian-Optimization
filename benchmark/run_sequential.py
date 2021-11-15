import os
import sys
from time import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

import bbobbenchmarks as bn
import fgeneric

np.random.seed(42)


class _GaussianProcessRegressor(GaussianProcessRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_fitted = False

    def fit(self, X, y):
        super().fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X, eval_MSE=False):
        _ = super().predict(X=X, return_std=eval_MSE)

        if eval_MSE:
            y_, sd = _
            sd2 = sd ** 2
            return y_, sd2
        else:
            return _


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
    sys.path.insert(0, "../../GaussianProcess")
    from BayesOpt import BO, DiscreteSpace, IntegerSpace, RandomForest, RealSpace
    from GaussianProcess import GaussianProcess
    from GaussianProcess.trend import constant_trend

    space = RealSpace([lb, ub]) * dim

    # kernel = 1.0 * Matern(length_scale=(1, 1), length_scale_bounds=(1e-10, 1e2))
    # model = _GaussianProcessRegressor(kernel=kernel, alpha=0, n_restarts_optimizer=30, normalize_y=False)

    mean = constant_trend(dim, beta=0)  # equivalent to Simple Kriging
    thetaL = 1e-5 * (ub - lb) * np.ones(dim)
    thetaU = 10 * (ub - lb) * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

    model = GaussianProcess(
        mean=mean,
        corr="matern",
        theta0=theta0,
        thetaL=thetaL,
        thetaU=thetaU,
        noise_estim=False,
        nugget=0,
        optimizer="BFGS",
        wait_iter=5,
        random_start=10 * dim,
        eval_budget=200 * dim,
    )

    return BO(
        search_space=space,
        obj_fun=obj_fun,
        model=model,
        DoE_size=dim * 10,
        max_FEs=max_FEs,
        verbose=True,
        n_point=1,
        minimize=True,
        acquisition_fun="EI",
        ftarget=ftarget,
        logger=None,
    )


if __name__ == "__main__":
    dims = (2,)
    fIDs = bn.nfreeIDs[6:]  # for all fcts
    instance = [1] * 10

    algorithm = test_BO

    opts = {"max_FEs": "50", "lb": -5, "ub": 5, "data_path": "./bbob_data/%s" % algorithm.__name__}
    opts["bbob_opt"] = {
        "comments": "max_FEs={0}".format(opts["max_FEs"]),
        "algid": algorithm.__name__,
    }

    for dim in dims:
        for fID in fIDs:
            for i in instance:
                run_optimizer(algorithm, dim, fID, i, logfile="./log", **opts)
