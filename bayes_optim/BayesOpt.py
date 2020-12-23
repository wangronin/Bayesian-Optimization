from typing import Callable, Any, Tuple, List, Optional
import os, sys, dill, functools, logging, time

import numpy as np
import json, copy, re
from copy import copy
from joblib import Parallel, delayed

from . import AcquisitionFunction
from .base import baseBO
from .Solution import Solution
from .SearchSpace import SearchSpace

__author__ = "Hao Wang"
__license__ = "3-clause BSD"


class BO(baseBO):
    def _create_acquisition(
        self,
        fun: Optional[Callable] = None,
        par: dict = {},
        return_dx: bool = False
    ):
        fun = fun if fun is not None else self._acquisition_fun
        if hasattr(getattr(AcquisitionFunction, fun), 'plugin'):
            if 'plugin' not in par:
                par.update({'plugin': self.fmin})

        return super()._create_acquisition(fun, par, return_dx)

    def pre_eval_check(self, X: Solution) -> Solution:
        """Check for the duplicated solutions as it is not allowed in noiseless cases
        """
        if not isinstance(X, Solution):
            X = Solution(X, var_name=self.var_names)

        N = X.N
        if hasattr(self, 'data'):
            X = X + self.data

        _ = []
        for i in range(N):
            x = X[i]
            idx = np.arange(len(X)) != i
            CON = np.all(
                np.isclose(
                    np.asarray(X[idx][:, self.r_index], dtype='float'),
                    np.asarray(x[self.r_index], dtype='float')
                ), axis=1
            )
            INT = np.all(X[idx][:, self.i_index] == x[self.i_index], axis=1)
            CAT = np.all(X[idx][:, self.d_index] == x[self.d_index], axis=1)
            if not any(CON & INT & CAT):
                _ += [i]

        return X[_]

class ParallelBO(BO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.n_point > 1

        if self._acquisition_fun == 'MGFI':
            self._par_name = 't'
            # Log-normal distribution for `t` supported on [0, \infty)
            self._sampler = lambda x: np.exp(np.log(x['t']) + 0.5 * np.random.randn())
        elif self._acquisition_fun == 'UCB':
            self._par_name = 'alpha'
            # Logit-normal distribution for `alpha` supported on [0, 1]
            self._sampler = lambda x: 1 / (1 + np.exp((x['alpha'] * 4 - 2) \
                + 0.6 * np.random.randn()))
        elif self._acquisition_fun == 'EpsilonPI':
            self._par_name = 'epsilon'
            self._sampler = None # TODO: implement this!
        else:
            raise NotImplementedError

        _criterion = getattr(AcquisitionFunction, self._acquisition_fun)()
        if self._par_name not in self._acquisition_par:
            self._acquisition_par[self._par_name] = getattr(_criterion, self._par_name)

    def _batch_arg_max_acquisition(
        self,
        n_point: int,
        return_dx: bool
    ):
        criteria = []
        for _ in range(n_point):
            _par = self._sampler(self._acquisition_par)
            _acquisition_par = copy(self._acquisition_par)
            _acquisition_par.update({self._par_name : _par})
            criteria.append(
                self._create_acquisition(par=_acquisition_par, return_dx=return_dx)
            )

        if self.n_job > 1:
            __ = Parallel(n_jobs=self.n_job)(
                delayed(self._argmax_restart)(c, logger=self._logger) for c in criteria
            )
        else:
            __ = [list(self._argmax_restart(_, logger=self._logger)) for _ in criteria]

        return tuple(zip(*__))

class AnnealingBO(ParallelBO):
    def __init__(
        self,
        t0: float = 2.,
        tf: float = 1e-1,
        schedule: str ='exp',
        *argv, **kwargs
    ):
        super().__init__(*argv, **kwargs)
        self.t0 = t0
        self.tf = tf
        self.schedule = schedule
        self._acquisition_par['t'] = t0

        # TODO: add supports for UCB and epsilon-PI
        max_iter = (self.max_FEs - self._DoE_size) / self.n_point
        if self.schedule == 'exp':                          # exponential
            self.alpha = (self.tf / t0) ** (1. / max_iter)
            self._anealer = lambda t: t * self.alpha
        elif self.schedule == 'linear':
            self.eta = (t0 - self.tf) / max_iter            # linear
            self._anealer = lambda t: t - self.eta
        elif self.schedule == 'log':
            self.c = self.tf * np.log(max_iter + 1)         # logarithmic
            self._anealer = lambda t: t * self.c / np.log(self.iter_count + 2)
        else:
            raise NotImplementedError

        def callback():
            self._acquisition_par['t'] = self._anealer(self._acquisition_par['t'])
        self._acquisition_callbacks += [callback]

class SelfAdaptiveBO(ParallelBO):
    def __init__(self, *argv, **kwargs):
        super.__init__(*argv, **kwargs)
        assert self.n_point > 1

    def _batch_arg_max_acquisition(
        self,
        n_point: int,
        return_dx: bool
    ):
        criteria = []
        _t_list = []
        N = int(n_point / 2)

        for _ in range(n_point):
            _t = np.exp(self._acquisition_par['t'] * np.random.randn())
            _t_list.append(_t)
            _acquisition_par = copy(self._acquisition_par)
            _acquisition_par.update({'t' : _t})
            criteria.append(
                self._create_acquisition(par=_acquisition_par, return_dx=return_dx)
            )

        if self.n_job > 1:
            __ = Parallel(n_jobs=self.n_job)(
                delayed(self._argmax_restart)(c, logger=self._logger) for c in criteria
            )
        else:
            __ = [list(self._argmax_restart(_, logger=self._logger)) for _ in criteria]

        # NOTE: this adaptation is different from what I did in the LeGO paper..
        idx = np.argsort(__[1])[::-1][:N]
        self._acquisition_par['t'] = np.mean([_t_list[i] for i in idx])
        return tuple(zip(*__))

    
class NoisyBO(ParallelBO):
    def pre_eval_check(self, X):
        if not isinstance(X, Solution):
            X = Solution(X, var_name=self.var_names)
        return X

    def _create_acquisition(self, fun=None, par={}, return_dx=False):
        if hasattr(getattr(AcquisitionFunction, self._acquisition_fun), 'plugin'):
            # use the model prediction to determine the plugin under noisy scenarios
            # TODO: add more options for determining the plugin value
            y_ = self.model.predict(self.data)
            plugin = np.min(y_) if self.minimize else np.max(y_)
            par.update({'plugin' : plugin})
        
        return super()._create_acquisition(par=par, return_dx=return_dx)
    

class IntensificationBO(ParallelBO):
    def __init__(
        self,
        max_r: int = 200,
        *argv,
        **kwargs
    ):
        """Bayesian Optimization for noisy objective functions where
        the re-sampling/evaluation of solutions is governed by the so-called
        "intensification" procedure proposed in

        [HutterHL11] Hutter, Frank, Holger H. Hoos, and Kevin Leyton-Brown.
            "Sequential model-based optimization for general algorithm configuration."
            In International conference on learning and intelligent optimization,
            pp. 507-523. Springer, Berlin, Heidelberg, 2011.

        Parameters
        ----------
        max_r : int, optional
            The maximal number of re-sampling allowed for the best-so-far solution,
            by default 200
        """
        super().__init__(*argv, **kwargs)
        self._max_r = max_r

    def pre_eval_check(self, X: Solution) -> Solution:
        if not isinstance(X, Solution):
            X = Solution(X, var_name=self.var_names)
        return X

    def evaluate(self, X: Solution) -> List:
        """The intensification procedure adopted from procedure 2 in [HutterHL11],
        where the consideration of instance/random seed is dropped. It iterates
        over each solution in `X` and compare the current solution to the incumbent
        by firstly adding one more re-evaluation to the incumbent solution and then
        successively performing 1, 2, 4, ... evaluations to the current solution
        until the current solution is worst than the incumbent, or the number of
        evaluations allocated to the current solution equals the incumbent.

        Parameters
        ----------
        X : Solution
            The candidate solutions to be evaluated

        Returns
        -------
        List
            The objective value for each solution in `X`
        """
        #Convert to internal representation to allow storing n_eval
        X = self._to_geno(X)
        #Need to explicitly check remaining budget to not exceed in intensification
        remaining_budget = self.max_FEs - self.eval_count
        print(f"rem: {remaining_budget}; max: {self.max_FEs}")
        if self.xopt is None:
            #First iteration, no incumbant yet
            incumbent = X[0]
        else:
            incumbent = self.xopt
            
        
        for x in X:
            # add one more sampling point to the incumbent
            if incumbent.n_eval < self._max_r:
                if remaining_budget > 0:
                    inc_pheno = self._to_pheno(incumbent)
                    if self._eval_type == 'dict':
                        inc_pheno = inc_pheno[0]
                    if incumbent.n_eval == 0:
                        #check for 0 since otherwise fitness keeps being nan
                        incumbent.fitness = self.obj_fun(inc_pheno)
                        incumbent.n_eval = 1
                        remaining_budget -= 1
                        self.eval_count += 1
                    else:
                        incumbent.fitness = (
                            incumbent.fitness * incumbent.n_eval + self.obj_fun(inc_pheno)
                        ) / (incumbent.n_eval + 1)
                        incumbent.n_eval += 1
                        remaining_budget -= 1
                        self.eval_count += 1
                else:
                    break

            N = 1
            while True:
                _N = min(N, incumbent.n_eval[0] - x.n_eval[0])
                _N = min(_N, remaining_budget)
                if _N > 0:
                    if self._eval_type == 'dict':
                        vals = [self.obj_fun(self._to_pheno(x)[0]) for _ in range(_N)]
                    else:
                        vals = [self.obj_fun(self._to_pheno(x)) for _ in range(_N)]
                    remaining_budget -= _N
                    self.eval_count += _N
                    print(f"new rem: {remaining_budget} ({self.max_FEs} - {self.eval_count})")
                    _val = np.sum(vals)
                    if x.n_eval == 0:
                        #check for 0 since otherwise fitness keeps being nan
                        x.fitness = _val
                        x.n_eval = _N
                    else:
                        x.fitness = (x.fitness * x.n_eval + _val) / (x.n_eval + _N)
                        x.n_eval += _N
                
                if self._compare(x.fitness, incumbent.fitness):
                    break
                elif _N == 0:
                    incumbent = x
                    break
                else:
                    N *= 2
            x.n_eval -= 1 #TODO: Fix this. Currently here because tell always adds 1
            self.eval_count -= 1 #TODO: Fix this. Currently here because tell always adds 1
        return X.fitness.tolist()

    def _create_acquisition(
        self,
        fun: Callable = None,
        par: dict = {},
        return_dx: bool = False
    ):
        if hasattr(getattr(AcquisitionFunction, self._acquisition_fun), 'plugin'):
            # use the model prediction to determine the plugin under noisy scenarios
            # TODO: add more options for determining the plugin value
            y_ = self.model.predict(self.data)
            plugin = np.min(y_) if self.minimize else np.max(y_)
            par.update({'plugin' : plugin})

        return super()._create_acquisition(par=par, return_dx=return_dx)