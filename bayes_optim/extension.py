import functools
from copy import copy
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.decomposition import PCA

from . import acquisition_fun as AcquisitionFunction
from .acquisition_fun import penalized_acquisition
from .bayes_opt import BO, ParallelBO
from .search_space import RealSpace, SearchSpace


class PCABO(BO):
    """Dimensionality reduction using Principle Component Decomposition (PCA)"""

    def __init__(self, kernel_pca: bool = False, n_components: Union[float, int] = None, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self._search_space, RealSpace)

        self.__search_space = self._search_space  # the original search space
        self.kernel_pca = kernel_pca
        self._n_components = n_components

    def _scale_Xy(
        self, new_X: List[List[float]], new_y: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(self, "data"):
            X = np.r_[self.data.astype(float), new_X]
            y = np.r_[self.data.fitness, new_y]
        else:
            X = np.atleast_2d(new_X).astype(float)
            y = np.array(new_y)

        self._X_mean = X.mean(axis=0)
        X -= self._X_mean
        y_ = -1 * y if not self.minimize else y

        r = rankdata(y_)
        N = len(y_)
        w = np.log(N) - np.log(r)
        w /= np.sum(w)
        return X * w.reshape(-1, 1), y

    def _compute_bounds(self, pca, search_space: SearchSpace):
        C = np.array([(l + u) / 2 for l, u in search_space.bounds])
        radius = np.sqrt(np.sum((np.array([l for l, _ in search_space.bounds]) - C) ** 2))
        C = C - pca.mean_ - self._X_mean
        C_ = C.dot(pca.components_.T)
        return [(_ - radius, _ + radius) for _ in C_]

    def _create_acquisition(self, fun=None, par=None, return_dx=False, **kwargs) -> Callable:
        par = {} if par is None else par
        acquisition_func = super()._create_acquisition(
            fun=fun, par=par, return_dx=return_dx, **kwargs
        )
        return functools.partial(
            penalized_acquisition,
            acquisition_func=acquisition_func,
            X_mean=self._X_mean,
            pca=self._pca,
            bounds=self.__search_space.bounds,
            return_dx=return_dx,
        )

    @property
    def xopt(self):
        if not hasattr(self, "data"):
            return None
        fopt = self._get_best(self.data.fitness)
        self._xopt = self.data[np.where(self.data.fitness == fopt)[0][0]]
        return self._xopt

    def ask(self, n_point: int = None) -> List[List[float]]:
        X = super().ask(n_point)
        if hasattr(self, "_pca"):
            X = self._pca.inverse_transform(X) + self._X_mean
        return X

    def tell(self, new_X, new_y):
        self.logger.info(f"observing {len(new_X)} points:")
        for i, x in enumerate(new_X):
            self.eval_count += 1
            self.logger.info(f"#{i + 1} - fitness: {new_y[i]}, solution: {x}")

        X, y = self._scale_Xy(new_X, new_y)
        if self.kernel_pca:
            # TODO: finish the kernel PCA part..
            # self._pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
            raise NotImplementedError
        else:
            self._pca = PCA(n_components=self._n_components, svd_solver="full")

        X = self._pca.fit_transform(X)
        bounds = self._compute_bounds(self._pca, self.__search_space)

        # set the search space in the reduced (feature) space
        self._search_space = RealSpace(bounds)
        X = self._to_geno(X)
        for i, _ in enumerate(X):
            X[i].fitness = y[i]
        self.data = self.post_eval_check(X)
        self.update_model()
        self.logger.info(f"xopt/fopt:\n{self.xopt}\n")


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
