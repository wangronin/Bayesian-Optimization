import logging, sys, functools
import numpy as np

from typing import Callable, Union
from copy import copy
from joblib import Parallel, delayed
from scipy.stats import rankdata, norm
from sklearn.decomposition import PCA, KernelPCA

from . import AcquisitionFunction
from .base import baseOptimizer
from .SearchSpace import ContinuousSpace
from .BayesOpt import BO, ParallelBO
from .misc import LoggerFormatter

def penalized_acquisition(x, acquisition_func, X_mean, pca, bounds, return_dx):
    x_ = pca.inverse_transform(x) + X_mean
    bounds = np.asarray(bounds)
    idx_lower = x_ < bounds[:, 0]
    idx_upper = x_ > bounds[:, 1]
    penalty = np.sum([bounds[i, 0] - x_[i] for i in idx_lower]) + \
        np.sum([x_[i] - bounds[i, 1] for i in idx_upper])
    penalty *= -1

    if penalty == 0:
        return acquisition_func(x)
    else:
        if return_dx:
            g = np.zeros((len(x), 1))
            g[idx_lower, :] = 1
            g[idx_upper, :] = -1
            return penalty, g
        else:
            return penalty

class PCABO(ParallelBO):
    def __init__(
        self,
        kernel_pca: bool = False,
        n_components: Union[float, int] = None,
        **kwargs
        ):
        super().__init__(**kwargs)
        assert isinstance(self._search_space, ContinuousSpace)

        self.__search_space = self._search_space # the original search space
        self.kernel_pca = kernel_pca
        self._n_components = n_components

    def _scale_X(self, X, func_vals):
        self._X_mean = X.mean(axis=0)
        X_ = X - self._X_mean

        if not self.minimize:
            func_vals = -1 * func_vals

        r = rankdata(func_vals)
        N = len(func_vals)
        w = np.log(N) - np.log(r)
        w /= np.sum(w)
        return X_ * w.reshape(-1, 1)

    def _compute_bounds(self, pca, search_space):
        C = np.array([(l + u) / 2 for l, u in search_space.bounds])
        radius = norm(np.np.array([l for l, _ in search_space.bounds]) - C)
        C = C - pca.mean_ - self._X_mean
        C_ = C.dot(pca.components_.T)
        return [(_ - radius, _ + radius) for _ in C_]

    def _create_acquisition(self, fun=None, par={}, return_dx=False):
        acquisition_func = super()._create_acquisition(fun, par, return_dx)
        fun = functools.partial(
            penalized_acquisition,
            acquisition_func=acquisition_func,
            X_mean=self._X_mean,
            pca=self._pca,
            bounds=self.__search_space.bounds,
            return_dx=return_dx
        )
        return fun

    def ask(self, n_point=None):
        X = super().ask(n_point)
        if hasattr(self, '_pca'):
            X = self._pca.inverse_transform(X) + self._X_mean
        return X

    def tell(self, X, func_vals):
        X_ = self._scale_X(X, func_vals)

        if self.kernel_pca:
            # TODO: finish the kernel PCA part..
            self._pca = KernelPCA(kernel='rbf', fit_inverse_transform=True, gamma=10)
        else:
            self._pca = PCA(n_components=self._n_components, svd_solver='full')

        X_ = self._pca.fit_transform(X_, func_vals)
        bounds = self._compute_bounds(self._pca, self.__search_space)

        # set the search space in the reduced (feature) space
        self._search_space = ContinuousSpace(bounds)
        super().tell(X_, func_vals)


class OptimizerPipeline(baseOptimizer):
    def __init__(
        self,
        obj_fun: Callable,
        ftarget: float = -np.inf,
        max_FEs: int = None,
        minimize: bool = True,
        verbose: bool = False,
        logger: str = None
        ):
        self.obj_fun = obj_fun
        self.max_FEs = max_FEs
        self.ftarget = ftarget
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
            return self.obj_fun(X)
        else:
            return [self.obj_fun(x) for x in X]

    def check_stop(self):
        return self._stop

def warm_start_pycma(BO):
    xopt = np.array(BO.xopt)
    fopt = np.array(BO.fopt)
    dim = BO.dim

    H = BO.model.Hessian(xopt)
    g = BO.model.gradient(xopt)[0]
    g /= np.linalg.norm(g)   #  normalize the gradient since its scale can be huge

    w, B = np.linalg.eigh(H)
    w[w <= 0] = 1e-6     # replace the negative eigenvalues by a very small value
    w_min, w_max = np.min(w), np.max(w)

    # to avoid the condition number gets too high
    cond_upper = 1e3
    delta = (cond_upper * w_min - w_max) / (1 - cond_upper)
    w += delta

    # the inverse transformation from the Hessian
    M = np.diag(1 / np.sqrt(w)).dot(B.T)
    H_inv = B.dot(np.diag(1 / w)).dot(B.T)
    p = -1 * H_inv.dot(g).ravel()
    alpha = np.linalg.norm(p)
    # sigma0 = np.linalg.norm(M.dot(g)) / np.sqrt(dim - 0.5)

    if np.isnan(alpha):
        alpha = 1
        H_inv = np.eye(dim)

    # use a backtracking line search to determine the initial step-size
    tau, c = 0.9, 1e-4
    slope = np.inner(g.ravel(), p.ravel())

    if slope > 0:  # this should not happen..
        p *= -1
        slope *= -1

    f = lambda x: BO.model.predict(x)
    while True:
        _x = (xopt + alpha * p).reshape(1, -1)
        if f(_x) <= f(xopt.reshape(1, -1)) + c * alpha * slope:
            break
        alpha *= tau

    sigma0 = np.linalg.norm(M.dot(alpha * p)) / np.sqrt(dim - 0.5)
    kwargs = {
        'x' : xopt,
        'fopt' : fopt,
        'sigma' : sigma0,
        'Cov' : H_inv,
    }
    return kwargs

class MultiAcquisitionBO(BO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.n_point > 1

        self._acquisition_fun = 'MGFI'
        self._acquisition_fun_list = ['MGFI', 'UCB']
        self._sampler_list = [
            lambda x: np.exp(np.log(x['t']) + 0.5 * np.random.randn()),
            lambda x: 1 / (1 + np.exp((x['alpha'] * 4 - 2) + 0.6 * np.random.randn()))
        ]
        self._par_name_list = ['t', 'alpha']
        self._acquisition_par_list = [{'t' : 1}, {'alpha' : 0.2}]
        self._N_acquisition = len(self._acquisition_fun_list)

        for i, _n in enumerate(self._par_name_list):
            _criterion = getattr(AcquisitionFunction, self._acquisition_fun_list[i])()
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
                delayed(self._argmax_restart)(c, logger=self._logger) for c in criteria
            )
        else:
            __ = [list(self._argmax_restart(_, logger=self._logger)) for _ in criteria]

        return tuple(zip(*__))


class ParallelBO2(ParallelBO):
    # TODO: add other Parallelization options:
    # 1) niching-based approach (my EVOLVE paper),
    # 2) bi-objective Pareto-front (PI vs. EI) (my WCCI '16 paper), and
    # 3) maybe QEI?
    pass


class RacingBO(ParallelBO):
    pass