import logging
import sys

import numpy as np

sys.path.insert(0, "../")
sys.path.insert(0, "../../pycma")

from copy import copy
from typing import Any, Callable, List, Tuple, Union

from bayes_optim import OptimizerPipeline, ParallelBO, RealSpace
from bayes_optim.surrogate import GaussianProcess, trend
from cma import CMAEvolutionStrategy, CMAOptions
from deap import benchmarks

np.random.seed(42)
Vector = List[float]
Matrix = List[Vector]


class _BO(ParallelBO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hist_EI = np.zeros(3)

    def ask(self, n_point=None):
        X = super().ask(n_point=n_point)
        if self.model.is_fitted:
            _criter = self._create_acquisition(fun="EI", par={}, return_dx=False)
            self._hist_EI[(self.iter_count - 1) % 3] = np.mean([_criter(x) for x in X])
        return X

    def check_stop(self):
        _delta = self._fBest_DoE - self.fopt
        if (
            self.iter_count > 1
            and np.mean(self._hist_EI[0 : min(3, self.iter_count - 1)]) < 0.01 * _delta
        ):
            self.stop_dict["low-EI"] = np.mean(self._hist_EI)

        if self.eval_count >= (self.max_FEs / 2):
            self.stop_dict["max_FEs"] = self.eval_count

        return super().check_stop()


class _CMA(CMAEvolutionStrategy):
    def __init__(
        self,
        dim: int,
        popsize: int,
        lb: Union[float, str, Vector, np.ndarray] = -np.inf,
        ub: Union[float, str, Vector, np.ndarray] = np.inf,
        ftarget: Union[int, float] = -np.inf,
        max_FEs: Union[int, str] = np.inf,
        verbose: bool = False,
        logger=None,
    ):

        inopts = {"bounds": [lb, ub], "ftarget": ftarget, "popsize": popsize}
        sigma0 = (ub - lb) / 5
        ub = np.array([ub] * dim)
        lb = np.array([lb] * dim)
        x0 = (ub - lb) * np.random.rand(dim) + lb

        super().__init__(x0=x0, sigma0=sigma0, inopts=inopts)
        self.dim = dim
        self.logger = logger
        self.max_FEs = max_FEs
        self.ftarget = ftarget
        self.verbose = verbose
        self.stop_dict = {}

    @property
    def eval_count(self):
        return self.countevals

    @property
    def iter_count(self):
        return self.countiter

    @property
    def x(self):
        return self.mean

    @x.setter
    def x(self, x):
        self.mean = copy(x)

    @property
    def Cov(self):
        return self.C

    @Cov.setter
    def Cov(self, C):
        try:
            w, B = np.linalg.eigh(C)
            if np.all(np.isreal(w)):
                self.B = B
                self.D = w ** 0.5
                self.dC = np.diag(C)
                self.C = C
        except np.linalg.LinAlgError:
            pass

    @property
    def logger(self):
        return self.logger

    @logger.setter
    def logger(self, logger):
        if isinstance(logger, logging.Logger):
            self.logger = logger
            self.logger.propagate = False
            return

    def ask(self, n_point=None):
        return super().ask(number=n_point)

    def tell(self, X, y):
        super().tell(X, y)
        x, f, _ = self.best.get()
        self.logger.info("iteration {}, fopt: {}, xopt: {}".format(self.countiter, f, x))

    def check_stop(self):
        _, f, __ = self.best.get()
        if f <= self.ftarget:
            self.stop_dict["ftarget"] = f

        if self.countevals >= self.max_FEs:
            self.stop_dict["FEs"] = self.countevals

        return bool(self.stop_dict)


dim = 2
n_point = 8
max_FEs = 30 * n_point
obj_fun = lambda x: benchmarks.himmelblau(x)[0]
lb, ub = -6, 6

search_space = RealSpace([lb, ub]) * dim
mean = trend.constant_trend(dim, beta=0)  # Ordinary Kriging

# autocorrelation parameters of GPR
thetaL = 1e-10 * (ub - lb) * np.ones(dim)
thetaU = 10 * (ub - lb) * np.ones(dim)
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

model = GaussianProcess(
    mean=mean,
    corr="squared_exponential",
    theta0=theta0,
    thetaL=thetaL,
    thetaU=thetaU,
    nugget=1e-5,
    noise_estim=False,
    optimizer="BFGS",
    wait_iter=5,
    random_start=5 * dim,
    eval_budget=100 * dim,
)

bo = _BO(
    search_space=search_space,
    obj_fun=obj_fun,
    model=model,
    eval_type="list",
    DoE_size=n_point,
    n_point=n_point,
    acquisition_fun="MGFI",
    acquisition_par={"t": 2},
    verbose=True,
    minimize=True,
)
cma = _CMA(dim=dim, popsize=n_point, lb=lb, ub=ub)


def post_BO(BO):
    xopt = np.array(BO.xopt)
    dim = BO.dim

    H = BO.model.Hessian(xopt)
    g = BO.model.gradient(xopt)[0]

    w, B = np.linalg.eigh(H)
    M = np.diag(1 / np.sqrt(w)).dot(B.T)
    H_inv = B.dot(np.diag(1 / w)).dot(B.T)
    sigma0 = np.linalg.norm(M.dot(g)) / np.sqrt(dim - 0.5)
    kwargs = {
        "x": xopt,
        "sigma": sigma0,
        "Cov": H_inv,
    }
    return kwargs


pipe = OptimizerPipeline(
    obj_fun=obj_fun, minimize=True, n_point=n_point, max_FEs=max_FEs, verbose=True
)
pipe.add(bo, transfer=post_BO)
pipe.add(cma)
pipe.run()
