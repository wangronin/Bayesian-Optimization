import copy
import functools
from collections import OrderedDict
from copy import copy
from typing import Callable, Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed

from . import acquisition_fun as AcquisitionFunction
from .acquisition_optim import argmax_restart
from .base import BaseBO
from .solution import Solution

__authors__ = ["Hao Wang"]


class BO(BaseBO):
    """The sequential Bayesian Optimization class"""

    def _create_acquisition(self, fun: Callable = None, par: Dict = None, **kwargv):
        fun = fun if fun is not None else self._acquisition_fun
        par = {} if par is None else par
        if hasattr(getattr(AcquisitionFunction, fun), "plugin"):
            if "plugin" not in par:
                par.update({"plugin": self.fmin if self.minimize else self.fmax})

        return super()._create_acquisition(fun, par, **kwargv)

    def pre_eval_check(self, X: List) -> List:
        """Check for the duplicated solutions as it is not allowed in noiseless cases"""
        if len(X) == 0:
            return X

        if not isinstance(X, Solution):
            X = Solution(X, var_name=self._search_space.var_name, n_obj=self.n_obj)

        N = X.N
        if hasattr(self, "data"):
            X = X + self.data

        _ = []
        for i in range(N):
            x = X[i]
            idx = np.arange(len(X)) != i
            CON = np.all(
                np.isclose(
                    np.asarray(X[idx][:, self.r_index], dtype="float"),
                    np.asarray(x[self.r_index], dtype="float"),
                ),
                axis=1,
            )
            INT = np.all(X[idx][:, self.i_index] == x[self.i_index], axis=1)
            CAT = np.all(X[idx][:, self.d_index] == x[self.d_index], axis=1)
            if not any(CON & INT & CAT):
                _ += [i]

        return X[_].tolist()


class ParallelBO(BO):
    """Batch-sequential Bayesian Optimization, which proposes multiple points in each iteration

    This class implements the multi-acquisition function approach, which samples multiple
    hyperparameter values for the acquisition function.

    """

    def __init__(
        self,
        n_point: int = 3,
        acquisition_fun: str = "MGFI",
        acquisition_par: Dict = {"t": 2},
        **kwargs,
    ):
        super().__init__(
            n_point=n_point,
            acquisition_fun=acquisition_fun,
            acquisition_par=acquisition_par,
            **kwargs,
        )
        assert self.n_point > 1

        if self._acquisition_fun == "MGFI":
            self._par_name = "t"
            # Log-normal distribution for `t` supported on [0, \infty)
            self._sampler = lambda x: np.exp(np.log(x["t"]) + 0.5 * np.random.randn())
        elif self._acquisition_fun == "UCB":
            self._par_name = "alpha"
            # Logit-normal distribution for `alpha` supported on [0, 1]
            self._sampler = lambda x: 1 / (1 + np.exp((x["alpha"] * 4 - 2) + 0.6 * np.random.randn()))
        elif self._acquisition_fun == "EpsilonPI":
            self._par_name = "epsilon"
            self._sampler = None  # TODO: implement this!
        else:
            raise NotImplementedError

        _criterion = getattr(AcquisitionFunction, self._acquisition_fun)()
        if self._par_name not in self._acquisition_par:
            self._acquisition_par[self._par_name] = getattr(_criterion, self._par_name)

    def _batch_arg_max_acquisition(self, n_point: int, return_dx: bool, fixed: Dict = None):
        criteria = []
        for _ in range(n_point):
            _par = self._sampler(self._acquisition_par)
            _acquisition_par = copy(self._acquisition_par)
            _acquisition_par.update({self._par_name: _par})
            criteria.append(self._create_acquisition(par=_acquisition_par, return_dx=return_dx, fixed=fixed))

        if self.n_job > 1:
            __ = Parallel(n_jobs=self.n_job)(
                delayed(self._argmax_restart)(c, logger=self.logger) for c in criteria
            )
        else:
            __ = [list(self._argmax_restart(_, logger=self.logger)) for _ in criteria]

        return tuple(zip(*__))


class AnnealingBO(ParallelBO):
    def __init__(self, t0: float = 2, tf: float = 1e-1, schedule: str = "exp", *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        self.t0 = t0
        self.tf = tf
        self.schedule = schedule
        self._acquisition_par["t"] = t0

        # TODO: add supports for UCB and epsilon-PI
        max_iter = (self.max_FEs - self._DoE_size) / self.n_point
        if self.schedule == "exp":  # exponential
            self.alpha = (self.tf / t0) ** (1.0 / max_iter)
            self._anealer = lambda t: t * self.alpha
        elif self.schedule == "linear":
            self.eta = (t0 - self.tf) / max_iter  # linear
            self._anealer = lambda t: t - self.eta
        elif self.schedule == "log":
            self.c = self.tf * np.log(max_iter + 1)  # logarithmic
            self._anealer = lambda t: t * self.c / np.log(self.iter_count + 2)
        else:
            raise NotImplementedError

        def callback():
            self._acquisition_par["t"] = self._anealer(self._acquisition_par["t"])

        self._acquisition_callbacks += [callback]


# TODO: write test file for this class
class SelfAdaptiveBO(ParallelBO):
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        assert self.n_point > 1

    def _batch_arg_max_acquisition(self, n_point: int, return_dx: bool, fixed: Dict = None):
        criteria = []
        _t_list = []
        N = int(n_point / 2)

        for _ in range(n_point):
            _t = np.exp(self._acquisition_par["t"] * np.random.randn())
            _t_list.append(_t)
            _acquisition_par = copy(self._acquisition_par)
            _acquisition_par.update({"t": _t})
            criteria.append(self._create_acquisition(par=_acquisition_par, return_dx=return_dx, fixed=fixed))

        if self.n_job > 1:
            __ = Parallel(n_jobs=self.n_job)(
                delayed(self._argmax_restart)(c, logger=self.logger) for c in criteria
            )
        else:
            __ = [list(self._argmax_restart(_, logger=self.logger)) for _ in criteria]

        # NOTE: this adaptation is different from what I did in the LeGO paper..
        idx = np.argsort(__[1])[::-1][:N]
        self._acquisition_par["t"] = np.mean([_t_list[i] for i in idx])
        return tuple(zip(*__))


class NoisyBO(ParallelBO):
    """Bayesian Optimization for Noisy Scenarios"""

    def pre_eval_check(self, X: List):
        if not isinstance(X, Solution):
            X = Solution(X, var_name=self.var_names)
        return X

    def _create_acquisition(self, fun: Callable = None, par: Dict = None, **kwargv):
        par = {} if par is None else par
        if hasattr(getattr(AcquisitionFunction, self._acquisition_fun), "plugin"):
            # use the model prediction to determine the plugin under noisy scenarios
            # TODO: add more options for determining the plugin value
            y_ = self.model.predict(self.data)
            plugin = np.min(y_) if self.minimize else np.max(y_)
            par.update({"plugin": plugin})

        return super()._create_acquisition(fun, par, **kwargv)


def partial_argument(func: callable, masks: np.ndarray, values: np.ndarray):
    """fill-in the values for inactive variables

    Parameters
    ----------
    func : callable
        the target function to call which is defined on the original search space
    masks : np.ndarray
        the mask array indicating which variables are deemed inactive
    values : np.ndarray
        the values fixed for the inactive variables
    """

    @functools.wraps(func)
    def wrapper(X):
        N = X.shape[1] if X.ndim > 1 else 1
        Y = np.empty((N, len(masks)))
        if len(values) > 0:
            Y[:, masks] = list(values)
        Y[:, ~masks] = X
        # Y = Y if X.ndim > 1 else Y[0]
        return func(Y)

    return wrapper


class NarrowingBO(BO):
    """BO with Automatic variable (de-)selection"""

    def __init__(
        self,
        var_selector: Callable = None,
        search_space_improving_fun: Callable = None,
        var_selection_FEs: int = 1,
        **kwargs,
    ):
        """Online automatic variable (de-)selection for BO

        Parameters
        ----------
        var_selector: Callable
            Function that rank the solution features, and returns a sub set of
            features to discard. The function will receive as input the data points
            evaluated and their fitness, the model, and the list of active features.
            Then, the function should return a dict of features and default value to deactivate.
        search_space_improving_fun: Callable
            Function that evaluates the overall performance of the narrowed search space. It should
            return a decision value (boolean, true if we should continue narrowing), and a dict with
            all the metrics. As input, the function receives the data, the model, and the previous
            metrics.
        var_selection_FEs: int
            Number of function evaluations before a (de)selection of features occurs
        """
        if "acquisition_optimization" in kwargs:
            kwargs["acquisition_optimization"]["optimizer"] = "MIES"
        else:
            kwargs["acquisition_optimization"] = {"optimizer": "MIES"}

        super().__init__(**kwargs)
        self.var_selector: callable = var_selector
        self.search_space_improving_fun: callable = search_space_improving_fun
        assert var_selection_FEs < self.max_FEs
        self.var_selection_FEs: int = var_selection_FEs
        self.inactive: OrderedDict = OrderedDict()

    def run(self):
        _metrics = {}
        while not self.check_stop():
            self.logger.info(f"iteration {self.iter_count} starts...")
            X = self.ask(fixed=self.inactive)
            func_vals = self.evaluate(X)
            self.tell(X, func_vals)

            if (self.eval_count % self.var_selection_FEs) == 0 and (
                self.DoE_size is None or self.eval_count > self.DoE_size
            ):
                decision, _metrics = self.search_space_improving_fun(self.data, self.model, _metrics)
                if decision:
                    inactive_var, value = self.var_selector(
                        self.data,
                        self.model,
                        set(self.search_space.var_name) - set(self.inactive.keys()),
                    )
                    self.inactive[inactive_var] = value
                    self.logger.info(f"fixing variable {inactive_var} to value {value}")
                else:
                    if len(self.inactive) != 0:
                        v = self.inactive.popitem()[0]
                        self.logger.info(f"adding variable {v} back to the search space")

        return self.xopt, self.fopt, self.stop_dict
