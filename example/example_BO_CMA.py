from pdb import set_trace

from typing import Callable, Any, Tuple
import os, sys, dill, functools, logging, time

import pandas as pd
import numpy as np
import json, copy, re 
from copy import copy
sys.path.insert(0, '../')

from BayesOpt import BO, SearchSpace, Solution
from BayesOpt.optimizer import OnePlusOne_CMA

class _BO(BO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hist_EI = np.zeros(3)

    def ask(self, n_point=None):
        X = super().ask(n_point=n_point)
        if self.model.is_fitted:
            _criter = self._create_acquisition(fun='EI', par={}, return_dx=False)
            self._hist_EI[self.iter_count % 3] = np.mean([_criter(x) for x in X])

        return X

    def check_stop(self):
        _delta = self._fBest_DoE - self.fopt
        if np.mean(self._hist_EI) < 0.01 * _delta:
            self.stop_dict['low-EI'] = np.mean(self._hist_EI)

        if self.eval_count >= (self.max_FEs / 2):
            self.stop_dict['max_FEs'] = self.eval_count

        return super().check_stop()


class OptimizerPipeline(object):
    def __init__(
        self,
        obj_fun: Callable,
        ftarget: float = None,
        max_FEs: int = None,
        minimize: bool = True,
        verbose: bool = False
        ):
        self.obj_fun = obj_fun
        self.max_FEs = max_FEs
        self.ftarget = ftarget
        self.verbose = verbose
        self.minimize = minimize
        self.queue = []
        self.stop_dict = []
        self.N = 0
        self._counter = 0
        self._curr_opt = None
        self._transfer = None
        self._stop = False

    def add(self, opt, transfer=None):
        opt.obj_fun = self.obj_fun
        opt.verbose = self.verbose
        opt.ftarget = self.ftarget
        opt.minimize = self.minimize
        opt.max_FEs = self.max_FEs
        self.queue.append((opt, transfer))
        self.N += 1

    def ask(self, n_point=1):
        """Get suggestions from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        if not self._curr_opt:
            self._curr_opt, self._transfer = self.queue[self._counter]

        return self._curr_opt.ask(n_point=n_point)

    def tell(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        # Update the model with new objective function observations
        # ...
        # No return statement needed
        self._curr_opt.tell(X, y)
        self.switch()
        self.xopt = self._curr_opt.xopt
        self.fopt = self._curr_opt.fopt

    def switch(self):
        # To check if the current optimizer is stopped
        # and switch to the next optimizer
        if self._curr_opt.check_stop():
            _max_FEs = self.max_FEs - self._curr_opt.eval_count
            _counter = self._counter + 1
            if _counter + 1 < self.N or _max_FEs > 0:
                _curr_opt, _transfer = self.queue[_counter]
                _curr_opt.max_FEs = _max_FEs
                self._counter = _counter

                if self._transfer:
                    kwargs = self._transfer(self._curr_opt)
                    for k, v in kwargs.items():
                        setattr(_curr_opt, k, v)

                self._curr_opt = _curr_opt
                self._transfer = _transfer
            else:
                self._stop = True
            
    def evaluate(self, X):
        return self._curr_opt.evaluate(X)

    def step(self):
        X = self.ask()    
        func_vals = self.evaluate(X)
        self.tell(X, func_vals)

    def run(self):
        while not self._stop:
            self.step()

        return self.xopt, self.fopt, self.stop_dict


if __name__ == '__main__':
    from deap import benchmarks
    from BayesOpt.SearchSpace import ContinuousSpace
    from GaussianProcess import GaussianProcess
    from GaussianProcess.trend import constant_trend

    np.random.seed(42)

    dim = 2
    max_FEs = 50
    obj_fun = lambda x: benchmarks.ackley(x)[0]
    lb, ub = -1, 6

    search_space = ContinuousSpace([lb, ub]) * dim
    mean = constant_trend(dim, beta=0)    

    # autocorrelation parameters of GPR
    thetaL = 1e-10 * (ub - lb) * np.ones(dim)
    thetaU = 10 * (ub - lb) * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

    model = GaussianProcess(
        mean=mean, corr='squared_exponential',
        theta0=theta0, thetaL=thetaL, thetaU=thetaU,
        nugget=0, noise_estim=False,
        optimizer='BFGS', wait_iter=5, random_start=dim,
        likelihood='concentrated', eval_budget=100 * dim
    )

    bo = _BO(
        search_space=search_space,
        obj_fun=obj_fun,
        model=model,
        eval_type='list',
        DoE_size=3,
        n_point=1,
        acquisition_fun='EI',
        verbose=True,
        minimize=True
    )
    cma = OnePlusOne_CMA(dim=dim, obj_fun=obj_fun, lb=lb, ub=ub)

    def post_BO(BO):
        xopt = BO.xopt.tolist()
        dim = BO.dim

        H = BO.model.Hessian(xopt)
        g = BO.model.gradient(xopt)[0]

        w, B = np.linalg.eigh(H)
        M = np.diag(1 / np.sqrt(w)).dot(B.T)
        H_inv = B.dot(np.diag(1 / 2)).dot(B.T)
        sigma0 = np.linalg.norm(M.dot(g)) / np.sqrt(dim - 0.5)

        kwargs = {
            'x0' : xopt,
            'sigma0' : sigma0,
            'C0' : H_inv,
        }
        return kwargs

    pipe = OptimizerPipeline(
        obj_fun=obj_fun, minimize=True, max_FEs=max_FEs, verbose=True
    )
    pipe.add(bo, transfer=post_BO)
    pipe.add(cma)
    pipe.run()