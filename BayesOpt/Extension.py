import logging, sys
import numpy as np

from typing import Callable
from copy import copy
from joblib import Parallel, delayed

from . import InfillCriteria
from .base import baseOptimizer
from .BayesOpt import BO
from .misc import LoggerFormatter

class OptimizerPipeline(baseOptimizer):
    def __init__(
        self,
        obj_fun: Callable,
        n_point: int = 1,
        ftarget: float = -np.inf,
        max_FEs: int = None,
        minimize: bool = True,
        verbose: bool = False,
        logger: str = None
        ):
        self.obj_fun = obj_fun
        self.max_FEs = max_FEs
        self.ftarget = ftarget
        self.n_point = n_point
        self.verbose = verbose
        self.minimize = minimize
        self.queue = []
        self.stop_dict = []
        self.N = 0
        self.logger = logger
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
        opt.logger = self._logger

        # add pairs of (optimizer, transfer function)
        self.queue.append((opt, transfer)) 
        self.N += 1

    def ask(self, n_point=None):
        """Get suggestions from the optimizer.

        Parameters
        ----------
        n_point : int
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
            self._logger.name = self._curr_opt.__class__.__name__
        
        n_point = n_point if n_point else self.n_point
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
        """To check if the current optimizer is stopped
        and switch to the next optimizer
        """
        if self._curr_opt.check_stop():
            _max_FEs = self.max_FEs - self._curr_opt.eval_count
            _counter = self._counter + 1

            if _counter < self.N and _max_FEs > 0:
                _curr_opt, _transfer = self.queue[_counter]
                _curr_opt.max_FEs = _max_FEs
                _curr_opt.xopt = np.array(self._curr_opt.xopt)
                _curr_opt.fopt = self._curr_opt.fopt

                if self._transfer:
                    kwargs = self._transfer(self._curr_opt)
                    for k, v in kwargs.items():
                        setattr(_curr_opt, k, v)

                self._logger.name = _curr_opt.__class__.__name__
                self._counter = _counter
                self._curr_opt = _curr_opt
                self._transfer = _transfer
            else:
                self._stop = True
            
    def evaluate(self, X):
        if not hasattr(X[0], '__iter__'):
            X = [X]
        return [self.obj_fun(x) for x in X]

class MultiAcquisitionBO(BO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.n_point > 1
        self._acquisition_fun = 'MGFI'  # TODO: this is for `self.update_model`

        # TODO: this is hard-wired. To make it generic
        self._acquisition_fun_list = ['MGFI', 'UCB']
        self._sampler_list = [
            lambda x: np.exp(np.log(x['t']) + 0.5 * np.random.randn()),
            lambda x: 1 / (1 + np.exp((x['alpha'] * 4 - 2) + 0.6 * np.random.randn())) 
        ]
        self._par_name_list = ['t', 'alpha']
        self._acquisition_par_list = [{'t' : 1}, {'alpha' : 0.2}]
        self._N_acquisition = len(self._acquisition_fun_list)

        for i, _n in enumerate(self._par_name_list):
            _criterion = getattr(InfillCriteria, self._acquisition_fun_list[i])()
            if _n not in self._acquisition_par_list[i]:
                self._acquisition_par_list[i][_n] = getattr(_criterion, _n)
        
    def _batch_arg_max_acquisition(self, n_point, return_dx):
        criteria = []
        
        for i in range(n_point):
            k = i % self._N_acquisition
            _acquisition_fun = self._acquisition_fun_list[k]
            _acquisition_par = self._acquisition_par_list[k]
            _par = self._sampler_list[k](_acquisition_par)
            _acquisition_par = copy(_acquisition_par)
            _acquisition_par.update({self._par_name_list[k] : _par})
            criteria.append(
                self._create_acquisition(
                    fun=_acquisition_fun, par=_acquisition_par, return_dx=return_dx
                )
            )
        
        if self.n_job > 1:
            __ = Parallel(n_jobs=self.n_job)(
                delayed(self._argmax_restart)(c) for c in criteria
            )
        else:
            __ = [list(self._argmax_restart(_)) for _ in criteria]
        
        return tuple(zip(*__))