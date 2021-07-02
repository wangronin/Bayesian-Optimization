import functools
from copy import copy
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from bayes_optim.solution import Solution

from . import acquisition_fun as AcquisitionFunction
from .acquisition_fun import penalized_acquisition
from .bayes_opt import BO, ParallelBO
from .search_space import RealSpace, SearchSpace
from .surrogate import GaussianProcess


class PCABO(BO):
    """Dimensionality reduction using Principle Component Decomposition (PCA)

    References

    [RaponiWB+20]
        Raponi, Elena, Hao Wang, Mariusz Bujny, Simonetta Boria, and Carola Doerr.
        "High dimensional bayesian optimization assisted by principal component analysis."
        In International Conference on Parallel Problem Solving from Nature, pp. 169-183.
        Springer, Cham, 2020.

    """

    def __init__(self, kernel_pca: bool = False, n_components: Union[float, int] = None, **kwargs):
        super().__init__(**kwargs)
        if self.model is not None:
            self.logger.warn(
                "The surrogate model will be created automatically by PCA-BO. "
                "The input argument `model` will be ignored"
            )
        assert isinstance(self._search_space, RealSpace)
        self.__search_space = self._search_space  # the original search space
        self.kernel_pca = kernel_pca  # whether to perform kernel PCA or not
        self._n_components = n_components  # the number of principal components or the

    def _get_scaled_Xy(self, data: Solution) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """center the data matrix and scale the data points with respect to the objective values

        Parameters
        ----------
        data : Solution
            the data matrix to scale

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            the scaled data matrix, the center data matrix, and the objective value
        """
        X, y = np.array(data), data.fitness
        self._X_mean = X.mean(axis=0)
        X -= self._X_mean
        y_ = -1 * y if not self.minimize else y

        r = rankdata(y_)
        N = len(y_)
        w = np.log(N) - np.log(r)
        w /= np.sum(w)
        return X * w.reshape(-1, 1), X, y

    def _compute_bounds(self, pca: PCA, search_space: SearchSpace) -> List[float]:
        C = np.array([(l + u) / 2 for l, u in search_space.bounds])
        radius = np.sqrt(np.sum((np.array([l for l, _ in search_space.bounds]) - C) ** 2))
        C = C - pca.mean_ - self._X_mean
        C_ = C.dot(pca.components_.T)
        return [(_ - radius, _ + radius) for _ in C_]

    def _create_acquisition(self, fun=None, par=None, return_dx=False, **kwargs) -> Callable:
        acquisition_func = super()._create_acquisition(
            fun=fun, par={} if par is None else par, return_dx=return_dx, **kwargs
        )
        # wrap the penalized acquisition function for handling the box constraints
        return functools.partial(
            penalized_acquisition,
            acquisition_func=acquisition_func,
            bounds=self.__search_space.bounds,  # hyperbox in the original space
            pca=self._pca,
            X_mean=self._X_mean,
            return_dx=return_dx,
        )

    def pre_eval_check(self, X: List) -> List:
        return X

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

        index = np.arange(len(new_X))
        if hasattr(self, "data"):
            index += len(self.data)

        new_X = self._to_geno(new_X, index)
        self.iter_count += 1
        for i, x in enumerate(new_X):
            self.eval_count += 1
            new_X[i].fitness = new_y[i]
            new_X[i].n_eval = 1
            self.logger.info(f"#{i + 1} - fitness: {new_y[i]}, solution: {x.tolist()}")

        new_X = self.post_eval_check(new_X)  # remove NaN's
        self.data = self.data + new_X if hasattr(self, "data") else new_X
        scaled_X, X, y = self._get_scaled_Xy(self.data)

        if self.kernel_pca:
            # TODO: finish the kernel PCA part..
            # self._pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
            raise NotImplementedError
        else:
            self._pca = PCA(n_components=self._n_components, svd_solver="full")

        self._pca.fit(scaled_X)  # re-fit the PCA transformation on the scaled data matrix
        X = self._pca.transform(X)  # transform the centered data matrix
        bounds = self._compute_bounds(self._pca, self.__search_space)
        # re-set the search space object for the reduced (feature) space
        self._search_space = RealSpace(bounds)
        # update the surrogate model
        self.update_model(X, y)
        self.logger.info(f"xopt/fopt:\n{self.xopt}\n")

    def update_model(self, X: np.ndarray, y: np.ndarray):
        # create the GPR model
        dim = self._search_space.dim
        bounds = np.array(self._search_space.bounds)
        _range = bounds[:, 1] - bounds[:, 0]
        thetaL, thetaU = (
            1e-8 * _range,
            10 * _range,
        )
        self.model = GaussianProcess(
            theta0=np.random.rand(dim) * (thetaU - thetaL) + thetaL,
            thetaL=thetaL,
            thetaU=thetaU,
            nugget=0,
            noise_estim=False,
            optimizer="BFGS",
            wait_iter=3,
            random_start=dim,
            likelihood="concentrated",
            eval_budget=100 * dim,
        )

        _std = np.std(y)
        y_ = y if np.isclose(_std, 0) else (y - np.mean(y)) / _std

        self.fmin, self.fmax = np.min(y_), np.max(y_)
        self.frange = self.fmax - self.fmin

        self.model.fit(X, y_)
        y_hat = self.model.predict(X)

        r2 = r2_score(y_, y_hat)
        MAPE = mean_absolute_percentage_error(y_, y_hat)
        self.logger.info(f"model r2: {r2}, MAPE: {MAPE}")


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
