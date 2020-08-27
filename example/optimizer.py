from pdb import set_trace

from typing import Callable, Any, Tuple
import os, sys, dill, functools, logging, time

import pandas as pd
import numpy as np
import json, copy, re 
from copy import copy

sys.path.insert(0, '../BayesOpt')

from BayesOpt import ParallelBO, SearchSpace, Solution
from BayesOpt.optimizer import cma_es

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

# default parameters of optimizers in the queue
_BO_par = {}
_CMA_par = {}

class _BO(ParallelBO):
    def __init__(self, **kwargs):
        self._hist_EI = np.zeros(3)

    def ask(self, n_point=None):
        X = super().ask(n_point=n_point)
        _criter = self._create_acquisition(fun='EI', par={}, return_dx=False)
        self._hist_EI[self.iter_count % 3] = np.mean([_criter(x) for x in X])

    def check_stop(self):
        _delta = self._fBest_DoE - self.fopt
        if np.mean(self._hist_EI) < 0.01 * _delta:
            self.stop_dict['EI'] = np.mean(self._hist_EI)

        return super().check_stop()

def post_BO(BO):
    xopt = BO.xopt
    dim = BO.dim

    H = BO._model.Hessian(xopt)
    g = BO._model.gradient(xopt)[0]

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

class BOCMA(AbstractOptimizer):
    def __init__(self, api_config):
        """Build wrapper class to use optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        self.opt_queue = iter([_BO, cma_es])
        self.transfer_func = iter([post_BO])
        self.default_pars = iter([_BO_par, _CMA_par])

        kwargs = next(self.default_pars)
        self._opt = next(self.opt_queue)(**kwargs)

    def suggest(self, n_suggestions=1):
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
        # Do whatever is needed to get the parallel guesses
        return self._opt.ask(n_point=n_suggestions)

    def observe(self, X, y):
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
        self._opt.tell(X, y)
        self.switch()

    def switch(self):
        # To check if the current optimizer is stopped
        # and switch to the next optimizer
        if self._opt.check_stop():
            try:
                kwargs = next(self.default_pars)
                kwargs.update(next(self.transfer_func)(self._opt))
                self._opt = next(self.opt_queue)(**kwargs)
            except:
                pass