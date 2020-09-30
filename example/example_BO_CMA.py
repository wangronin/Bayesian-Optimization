from pdb import set_trace

import os, sys, time
import numpy as np
sys.path.insert(0, '../')

from deap import benchmarks
from bayes_optim import OptimizerPipeline, BO, Solution, ContinuousSpace
from bayes_optim.acquisition_optim import OnePlusOne_Cholesky_CMA
from bayes_optim.Surrogate import GaussianProcess, trend

class _BO(BO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hist_EI = np.zeros(3)

    def ask(self, n_point=None):
        X = super().ask(n_point=n_point)
        if self.model.is_fitted:
            _criter = self._create_acquisition(fun='EI', par={}, return_dx=False)
            self._hist_EI[(self.iter_count - 1) % 3] = np.mean([_criter(x) for x in X])
        return X

    def check_stop(self):
        _delta = self._fBest_DoE - self.fopt
        if self.iter_count > 1 and \
            np.mean(self._hist_EI[0:min(3, self.iter_count - 1)]) < 0.01 * _delta:
            self.stop_dict['low-EI'] = np.mean(self._hist_EI)

        if self.eval_count >= (self.max_FEs / 2):
            self.stop_dict['max_FEs'] = self.eval_count

        return super().check_stop()


np.random.seed(42)
dim = 2
max_FEs = 40
obj_fun = lambda x: benchmarks.himmelblau(x)[0]
lb, ub = -6, 6

search_space = ContinuousSpace([lb, ub]) * dim
mean = trend.constant_trend(dim, beta=None)    

# autocorrelation parameters of GPR
thetaL = 1e-10 * (ub - lb) * np.ones(dim) / (ub - lb) ** 2
thetaU = 10 * np.ones(dim) / (ub - lb) ** 2
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

model = GaussianProcess(
    mean=mean, corr='squared_exponential',
    theta0=theta0, thetaL=thetaL, thetaU=thetaU,
    nugget=1e-5, noise_estim=False,
    optimizer='BFGS', wait_iter=5, random_start=5 * dim,
    eval_budget=100 * dim
)

bo = _BO(
    search_space=search_space,
    obj_fun=obj_fun,
    model=model,
    eval_type='list',
    DoE_size=10,
    n_point=1,
    acquisition_fun='EI',
    verbose=True,
    minimize=True
)
cma = OnePlusOne_Cholesky_CMA(dim=dim, obj_fun=obj_fun, lb=lb, ub=ub)

def post_BO(BO):
    xopt = np.array(BO.xopt)
    dim = BO.dim

    H = BO.model.Hessian(xopt)
    g = BO.model.gradient(xopt)[0]

    w, B = np.linalg.eigh(H)
    w[w <= 0] = 1e-6     # replace the negative eigenvalues by a very small value
    w_min, w_max = np.min(w), np.max(w)

    # to avoid the conditional number gets too high
    cond_upper = 1e3
    delta = (cond_upper * w_min - w_max) / (1 - cond_upper)
    w += delta

    M = np.diag(1 / np.sqrt(w)).dot(B.T)
    H_inv = B.dot(np.diag(1 / w)).dot(B.T)
    sigma0 = np.linalg.norm(M.dot(g)) / np.sqrt(dim - 0.5)

    # use a backtracking line search to determine the initial step-size
    tau, c = 0.9, .1
    p = -1 * H_inv.dot(g).ravel()
    slope = np.inner(g.ravel(), p.ravel())

    if slope > 0:  # this should not happen..
        p *= -1
        slope *= -1

    f = lambda x: BO.model.predict(x)
    while True:
        _x = (xopt + sigma0 * p).reshape(1, -1)
        if f(_x) <= f(xopt.reshape(1, -1)) + c * sigma0 * slope:
            break
        sigma0 *= tau

    if sigma0 == 0:
        sigma0 = 1 / 5

    if np.isnan(sigma0):
        sigma0 = 1 / 5
        H_inv = np.eye(dim)
    
    kwargs = {
        'x' : xopt,
        'sigma' : sigma0,
        'Cov' : H_inv,
    }
    return kwargs

pipe = OptimizerPipeline(
    obj_fun=obj_fun, minimize=True, max_FEs=max_FEs, verbose=True
)
pipe.add(bo, transfer=post_BO)
pipe.add(cma)
pipe.run()