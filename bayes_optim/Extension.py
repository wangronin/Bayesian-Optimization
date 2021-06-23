import functools
from copy import copy
from typing import Dict, Union

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm, rankdata
from sklearn.decomposition import PCA, KernelPCA

from . import acquisition_fun as AcquisitionFunction
from .acquisition_fun import penalized_acquisition
from .bayes_opt import ParallelBO
from .search_space import RealSpace


class PCABO(ParallelBO):
    """Dimensionality reduction using Principle Component Decomposition (PCA)"""

    def __init__(self, kernel_pca: bool = False, n_components: Union[float, int] = None, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self._search_space, RealSpace)

        self.__search_space = self._search_space  # the original search space
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
            return_dx=return_dx,
        )
        return fun

    def ask(self, n_point=None):
        X = super().ask(n_point)
        if hasattr(self, "_pca"):
            X = self._pca.inverse_transform(X) + self._X_mean
        return X

    def tell(self, X, func_vals):
        X_ = self._scale_X(X, func_vals)

        if self.kernel_pca:
            # TODO: finish the kernel PCA part..
            self._pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
        else:
            self._pca = PCA(n_components=self._n_components, svd_solver="full")

        X_ = self._pca.fit_transform(X_, func_vals)
        bounds = self._compute_bounds(self._pca, self.__search_space)

        # set the search space in the reduced (feature) space
        self._search_space = RealSpace(bounds)
        super().tell(X_, func_vals)


class MultiAcquisitionBO(ParallelBO):
    """Using multiple acquisition functions for parallelization"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.n_point > 1

        self._acquisition_fun = "MGFI"
        self._acquisition_fun_list = ["MGFI", "UCB"]
        self._sampler_list = [
            lambda x: np.exp(np.log(x["t"]) + 0.5 * np.random.randn()),
            lambda x: 1 / (1 + np.exp((x["alpha"] * 4 - 2) + 0.6 * np.random.randn())),
        ]
        self._par_name_list = ["t", "alpha"]
        self._acquisition_par_list = [{"t": 1}, {"alpha": 0.1}]
        self._N_acquisition = len(self._acquisition_fun_list)

        for i, _n in enumerate(self._par_name_list):
            _criterion = getattr(AcquisitionFunction, self._acquisition_fun_list[i])()
            if _n not in self._acquisition_par_list[i]:
                self._acquisition_par_list[i][_n] = getattr(_criterion, _n)

    def _batch_arg_max_acquisition(self, n_point: int, return_dx: bool, fixed: Dict = None):
        criteria = []
        for i in range(n_point):
            k = i % self._N_acquisition
            _acquisition_fun = self._acquisition_fun_list[k]
            _acquisition_par = self._acquisition_par_list[k]
            _par = self._sampler_list[k](_acquisition_par)
            _acquisition_par = copy(_acquisition_par)
            _acquisition_par.update({self._par_name_list[k]: _par})
            criteria.append(
                self._create_acquisition(
                    fun=_acquisition_fun, par=_acquisition_par, return_dx=return_dx, fixed=fixed
                )
            )

        if self.n_job > 1:
            __ = Parallel(n_jobs=self.n_job)(
                delayed(self._argmax_restart)(c, logger=self.logger) for c in criteria
            )
        else:
            __ = [list(self._argmax_restart(_, logger=self.logger)) for _ in criteria]

        return tuple(zip(*__))
