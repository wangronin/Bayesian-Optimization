from pdb import set_trace

import os, sys, time
import numpy as np
sys.path.insert(0, '../')

from deap import benchmarks
from bayes_optim import OptimizerPipeline, BO, Solution, ContinuousSpace
from bayes_optim.acquisition_optim import OnePlusOne_Cholesky_CMA
from bayes_optim.Surrogate import GaussianProcess, trend
from bayes_optim.Extension import warm_start_pycma

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


np.random.seed(666)
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

pipe = OptimizerPipeline(
    obj_fun=obj_fun, minimize=True, max_FEs=max_FEs, verbose=True
)
pipe.add(bo, transfer=warm_start_pycma)
pipe.add(cma)
pipe.run()