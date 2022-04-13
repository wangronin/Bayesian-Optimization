from __future__ import annotations

import functools
from copy import copy, deepcopy
from typing import Callable, Dict, List, Union
from enum import Enum

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from abc import ABC, abstractmethod
from sklearn.base import clone, is_regressor

from .acquisition import acquisition_fun as AcquisitionFunction
from .acquisition.optim import OptimizationListener
from .bayes_opt import BO, ParallelBO
from .search_space import RealSpace, SearchSpace
from .solution import Solution
from .surrogate import GaussianProcess, RandomForest, trend
from .utils import timeit
from .kpca import MyKernelPCA, create_kernel
from .utils import partial_argument

from .mylogging import *
import time
import bisect
import scipy
from collections import deque


GLOBAL_CHARTS_SAVER = None


class LinearTransform(PCA):
    def __init__(self, minimize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.minimize = minimize

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearTransform:
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
        self.center = X.mean(axis=0)
        X_centered = X - self.center
        y_ = -1 * y if not self.minimize else y
        r = rankdata(y_)
        N = len(y_)
        w = np.log(N) - np.log(r)
        w /= np.sum(w)
        X_scaled = X_centered * w.reshape(-1, 1)
        return super().fit(X_scaled)  # fit the PCA transformation on the scaled data matrix

    def transform(self, X: np.ndarray) -> np.ndarray:
        return super().transform(X - self.center)  # transform the centered data matrix

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "components_"):
            return X
        inversed = super().inverse_transform(X) + self.center
        eprintf("Restored point is", inversed)
        return inversed

    def get_explained_variance_ratio(self):
        return super().explained_variance_ratio_


KernelFitStrategy = Enum('KernelFitStrategy', 'AUTO FIXED_KERNEL LIST_OF_KERNEL')


class KernelTransform(MyKernelPCA):
    def __init__(self, dimensions: int, minimize: bool = True, kernel_fit_strategy: KernelFitStrategy = KernelFitStrategy.AUTO, kernel_config: dict = None, **kwargs):
        super().__init__(kernel_config=kernel_config, dimensions=dimensions, **kwargs)
        self.minimize = minimize
        self.kernel_fit_strategy = kernel_fit_strategy
        self.kernel_config = kernel_config
        self.N_same_kernel = 1
        self.__count_same_kernel = self.N_same_kernel
        self.__is_kernel_refit_needed = True

    @staticmethod
    def _check_input(X):
        return np.atleast_2d(np.asarray(X, dtype=float))

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # eprintf(f"My kernel parameters {self.kernel_config}")
        # eprintf(f"{self.fit_transform.__name__}: Fit X")
        # eprintf(X.tolist())
        # eprintf(f"{self.fit_transform.__name__}: Fit y")
        # eprintf(y.tolist())
        X = self._check_input(X)
        self.fit(X, y)
        transformed = self.transform(X)
        # eprintf(f"Transformed")
        # eprintf(transformed.tolist())
        return transformed

    def fit(self, X: np.ndarray, y: np.ndarray) -> KernelTransform:
        X_scaled = self._weighting_scheme(X, y)
        if self.__is_kernel_refit_needed:
            self.__fit_kernel_parameters(X_scaled, KernelParamsOptimizerSearch)
        return super().fit(X_scaled)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._check_input(X)
        return super().transform(X)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        Y = self._check_input(Y)
        return super().inverse_transform(Y)

    def set_kernel_refit_needed(self, is_needed):
        self.__is_kernel_refit_needed = is_needed

    def get_kernel_parameters(self):
        return self.kernel_config

    def _weighting_scheme(self, X, y):
        X = self._check_input(X)
        self.center = X.mean(axis=0)
        X_centered = X - self.center
        y_ = -1 * y if not self.minimize else y
        r = rankdata(y_)
        N = len(y_)
        w = np.log(N) - np.log(r)
        w /= np.sum(w)
        X_scaled = X_centered * w.reshape(-1, 1)
        return X_scaled

    def __fit_kernel_parameters(self, X, SearchStrategy):
        self.__is_kernel_refit_needed = False
        if self.kernel_fit_strategy is KernelFitStrategy.FIXED_KERNEL:
            pass
        elif self.kernel_fit_strategy is KernelFitStrategy.AUTO:
            kernel_name, kernel_parameters, result = SearchStrategy(self.X_initial_space, X, self.epsilon).find_best_kernel(['rbf'])
            self.kernel_config = {'kernel_name': kernel_name, 'kernel_parameters': kernel_parameters}
        elif self.kernel_fit_strategy is KernelFitStrategy.LIST_OF_KERNEL:
            kernel_name, kernel_parameters, result = SearchStrategy(self.X_initial_space, X, self.epsilon).find_best_kernel(self.kernel_config['kernel_names'])
            self.kernel_config = {'kernel_name': kernel_name, 'kernel_parameters': kernel_parameters}
        else:
            raise ValueError(f'Kernel fit strategy {str(KernelFitStrategy)} is not known')


class KernelParamsSearchStrategy(ABC):
    def __init__(self, X: np.ndarray, X_weighted: np.ndarray, epsilon: float):
        self._X = X
        self._X_weighted = X_weighted
        self._epsilon = epsilon

    def find_best_kernel(self, kernel_names: List):
        best_result = -10
        best_kernel, best_kernel_name, best_kernel_params = None, None, None
        for kernel_name in kernel_names:
            kernel_name, kernel_params, result = self.find_best_for_kernel(kernel_name)
            if result > best_result:
                best_result = result
                best_kernel_name, best_kernel_params = kernel_name, kernel_params
        return best_kernel_name, best_kernel_params, best_result

    def find_best_for_kernel(self, kernel_name: str):
        if kernel_name == 'rbf':
            params, result = self.find_best_for_rbf()
        elif kernel_name == 'poly':
            params, result = self.find_best_for_poly()
        else:
            raise NotImplementedError
        return kernel_name, params, result

    @staticmethod
    def __try_kernel(parameters, epsilon, X_initial_space, X_weighted, kernel_name):
        kpca = MyKernelPCA(epsilon, X_initial_space, {'kernel_name': kernel_name, 'kernel_parameters': parameters})
        kpca.fit(X_weighted)
        if kpca.too_compressed:
            return int(1e9), 0
        return kpca.k, kpca.extracted_information

    @staticmethod
    def __try_rbf_kernel(parameters, epsilon, X_initial_space, X_weighted, kernel_name, l2_norm_matrix):
        kpca = MyKernelPCA(epsilon, X_initial_space, {'kernel_name': kernel_name, 'kernel_parameters': parameters})
        kpca._set_reuse_rbf_data(l2_norm_matrix)
        kpca.fit(X_weighted)
        if kpca.too_compressed:
            return int(1e9), 0
        return kpca.k, kpca.extracted_information

    @abstractmethod
    def _optimize_rbf_gamma(f):
        pass

    def find_best_for_rbf(self):
        self.__l2_norm_matrix = [[np.sum((np.array(a) - np.array(b)) ** 2) for a in self._X_weighted] for b in self._X_weighted]
        f = functools.partial(KernelParamsSearchStrategy.__try_rbf_kernel, epsilon=self._epsilon, X_initial_space=self._X, X_weighted=self._X_weighted, kernel_name='rbf', l2_norm_matrix=self.__l2_norm_matrix)
        return self._optimize_rbf_gamma(f)

    def find_best_for_poly(self):
        raise NotImplementedError


class KernelParamsGridSearch(KernelParamsSearchStrategy):
    @staticmethod
    def __gamma_exponential_grid_minimizer(f, start, end, steps):
        t1 = math.log(start)
        t2 = math.log(end)
        eps = (t2 - t1) / 100
        mi, max_second_value, optimal_parameter = int(1e9), int(-1e9), 0.
        for i in range(steps):
            gamma = math.pow(math.e, t1 + i*eps)
            first_value, second_value = f({'gamma': gamma})
            if math.isnan(first_value) or math.isnan(second_value):
                continue
            if first_value == mi and second_value > max_second_value:
                max_second_value = second_value
                optimal_parameter = gamma
            if first_value < mi:
                mi = first_value
                max_second_value = second_value
                optimal_parameter = gamma
        eprintf(f"Min dimensionality is {mi}, with max extracted information {max_second_value} and parameters {optimal_parameter}")
        return {'gamma': optimal_parameter}, mi

    @staticmethod
    def __gamma_sequential_grid_minimizer(f, start, end, steps):
        step_size = (end - start) / 100
        mi, max_second_value, optimal_parameter = int(1e9), int(-1e9), 0.
        for i in range(steps):
            gamma = start + i * step_size
            first_value, second_value = f({'gamma': gamma})
            if math.isnan(first_value) or math.isnan(second_value):
                continue
            if first_value == mi and second_value > max_second_value:
                max_second_value = second_value
                optimal_parameter = gamma
            if first_value < mi:
                mi = first_value
                max_second_value = second_value
                optimal_parameter = gamma
        eprintf(f"Min dimensionality is {mi}, with max extracted information {max_second_value} and parameters {optimal_parameter}")
        return {'gamma': optimal_parameter}, mi

    def _optimize_rbf_gamma(self, f):
        return KernelParamsGridSearch.__gamma_sequential_grid_minimizer(f, 1e-4, 1, 100)


class KernelParamsOptimizerSearch(KernelParamsSearchStrategy):
    @staticmethod
    def __f_wrapper(f, x):
        a, b = f({'gamma': x[0]})
        return float(a - b)

    def _get_componenets_and_info(self, fvalue):
        min_components = int(math.floor(fvalue)) + 1
        info = min_components - fvalue
        return min_components, info

    def __opt(self, f):
        res = scipy.optimize.minimize(f, method='L-BFGS-B', x0=[0.05], bounds=[(1e-4, 2.)], options={'maxiter':200})
        gamma = res.x[0]
        fopt, info = self._get_componenets_and_info(res.fun)
        eprintf(f"Min dimensionality is {fopt}, with extracted information {info} and parameters gamma={gamma}")
        return {'gamma': gamma}, fopt

    def _optimize_rbf_gamma(self, f):
        return self.__opt(functools.partial(KernelParamsOptimizerSearch.__f_wrapper, f))


class KernelParamsCombinedSearch(KernelParamsOptimizerSearch):
    @staticmethod
    def __f_wrapper(f, x):
        a, b = f({'gamma': x[0]})
        return float(a - b)

    def __opt(self, f, initial, lb, ub):
        res = scipy.optimize.minimize(f, method='L-BFGS-B', x0=[initial], bounds=[(lb, ub)], options={'maxiter':100})
        gamma = res.x[0]
        fopt, info = self._get_componenets_and_info(res.fun)
        eprintf(f"Min dimensionality is {fopt}, with extracted information {info} and parameters gamma={gamma}")
        return {'gamma': gamma}, fopt

    def __grid_search(self, f, begin, step_size, evals):
        mi = float('inf')
        arg_mi = 0.
        for i in range(evals):
            gamma = begin + i * step_size
            cur = f([gamma])
            if cur < mi:
                mi = cur
                arg_mi = gamma
        return arg_mi

    def _optimize_rbf_gamma(self, f):
        f = functools.partial(KernelParamsCombinedSearch.__f_wrapper, f)
        begin, step_size, evals = 0.01, 0.009, 100
        x0 = self.__grid_search(f, begin, step_size, evals)
        return self.__opt(f, x0, x0 - step_size, x0 + step_size)


def penalized_acquisition(x, acquisition_func, bounds, pca, return_dx):
    bounds_ = np.atleast_2d(bounds)
    # map back the candidate point to check if it falls inside the original domain
    x_ = pca.inverse_transform(x).ravel()
    eprintf("Bounds", bounds_)
    idx_lower = np.where(x_ < bounds_[:, 0])[0]
    idx_upper = np.where(x_ > bounds_[:, 1])[0]
    penalty = -1 * (
        np.sum([bounds_[i, 0] - x_[i] for i in idx_lower])
        + np.sum([x_[i] - bounds_[i, 1] for i in idx_upper])
    )
    eprintf("Penalty", penalty)

    if penalty == 0:
        out = acquisition_func(x)
    else:
        if return_dx:
            # gradient of the penalty in the original space
            g_ = np.zeros((len(x_), 1))
            g_[idx_lower, :] = 1
            g_[idx_upper, :] = -1
            # get the gradient of the penalty in the reduced space
            g = pca.components_.dot(g_)
            out = (penalty, g)
        else:
            out = penalty
    eprintf("Penalized acq function value", out)
    return out


class PCABO(BO):
    """Dimensionality reduction using Principle Component Decomposition (PCA)

    References

    [RaponiWB+20]
        Raponi, Elena, Hao Wang, Mariusz Bujny, Simonetta Boria, and Carola Doerr.
        "High dimensional bayesian optimization assisted by principal component analysis."
        In International Conference on Parallel Problem Solving from Nature, pp. 169-183.
        Springer, Cham, 2020.

    """

    def __init__(self, n_components: Union[float, int] = None, **kwargs):
        super().__init__(**kwargs)
        if self.model is not None:
            self.logger.warning(
                "The surrogate model will be created automatically by PCA-BO. "
                "The input argument `model` will be ignored"
            )
        assert isinstance(self._search_space, RealSpace)
        self.__search_space = deepcopy(self._search_space)  # the original search space
        self._pca = LinearTransform(n_components=n_components, svd_solver="full", minimize=self.minimize)
        global GLOBAL_CHARTS_SAVER
        GLOBAL_CHARTS_SAVER = MyChartSaver('PCABO-1', 'PCABO', self._search_space.bounds, self.obj_fun)
        self.acq_opt_time = 0
        self.mode_fit_time = 0

    @staticmethod
    def _compute_bounds(pca: PCA, search_space: SearchSpace) -> List[float]:
        C = np.array([(l + u) / 2 for l, u in search_space.bounds])
        radius = np.sqrt(np.sum((np.array([l for l, _ in search_space.bounds]) - C) ** 2))
        C = C - pca.mean_ - pca.center
        C_ = C.dot(pca.components_.T)
        return [(_ - radius, _ + radius) for _ in C_]

    def get_lower_space_dimensionality(self):
        return self.search_space.dim

    def get_extracted_information(self):
        if hasattr(self._pca, 'explained_variance_ratio_'):
            return sum(self._pca.explained_variance_ratio_)
        return None

    def _create_acquisition(self, fun=None, par=None, return_dx=False, **kwargs) -> Callable:
        acquisition_func = super()._create_acquisition(
            fun=fun, par={} if par is None else par, return_dx=return_dx, **kwargs
        )
        # TODO: make this more general for other acquisition functions
        # wrap the penalized acquisition function for handling the box constraints
        return functools.partial(
            penalized_acquisition,
            acquisition_func=acquisition_func[0],
            bounds=self.__search_space.bounds,  # hyperbox in the original space
            pca=self._pca,
            return_dx=return_dx,
            ), functools.partial(
                penalized_acquisition,
                acquisition_func=acquisition_func[1],
                bounds=self.__search_space.bounds,  # hyperbox in the original space
                pca=self._pca,
                return_dx=False,
            )

    def pre_eval_check(self, X: List) -> List:
        """Note that we do not check against duplicated point in PCA-BO since those points are
        sampled in the reduced search space. Please check if this is intended
        """
        if isinstance(X, np.ndarray):
            X = X.tolist()
        return X

    @property
    def xopt(self):
        if not hasattr(self, "data"):
            return None
        fopt = self._get_best(self.data.fitness)
        self._xopt = self.data[np.where(self.data.fitness == fopt)[0][0]]
        return self._xopt

    def ask(self, n_point: int = None) -> List[List[float]]:
        start = time.time()
        new_x1 = self._pca.inverse_transform(super().ask(n_point))
        self.acq_opt_time = time.time() - start
        new_x = new_x1[0]
        is_inside = sum(1 for i in range(len(new_x)) if self.__search_space.bounds[i][0]<= new_x[i] <= self.__search_space.bounds[i][1])==len(new_x)
        eprintf("Is inside?", is_inside)
        if type(new_x1) is np.ndarray:
            new_x1 = new_x1.tolist()
        return new_x1

    def tell(self, new_X, new_y):
        self.logger.info(f"observing {len(new_X)} points:")
        for i, x in enumerate(new_X):
            self.logger.info(f"#{i + 1} - fitness: {new_y[i]}, solution: {x}")

        index = np.arange(len(new_X))
        if hasattr(self, "data"):
            index += len(self.data)
        # convert `new_X` to a `Solution` object
        new_X = self._to_geno(new_X, index=index, n_eval=1, fitness=new_y)
        self.iter_count += 1
        self.eval_count += len(new_X)
        GLOBAL_CHARTS_SAVER.set_iter_number(self.iter_count)

        new_X = self.post_eval_check(new_X)  # remove NaN's
        self.data = self.data + new_X if hasattr(self, "data") else new_X
        # re-fit the PCA transformation
        X = self._pca.fit_transform(np.array(self.data), self.data.fitness)
        # GLOBAL_CHARTS_SAVER.save_with_manifold(self.iter_count, self.data, X, self._pca)

        # update the surrogate model
        self.update_model(X, self.data.fitness)
        self.logger.info(f"xopt/fopt:\n{self.xopt}\n")

    def update_model(self, X: np.ndarray, y: np.ndarray):
        # NOTE: the GPR model will be created since the effective search space (the reduced space
        # is dynamic)
        bounds = self._compute_bounds(self._pca, self.__search_space)
        self._search_space = RealSpace(bounds)
        bounds = np.asarray(bounds)
        dim = self._search_space.dim
        self.model = self.create_default_model(self.search_space, self.my_seed)
        _std = np.std(y)
        y_ = y
        self.fmin, self.fmax = np.min(y_), np.max(y_)
        self.frange = self.fmax - self.fmin

        start = time.time()
        self.model.fit(X, y_)
        self.mode_fit_time = time.time() - start
        GLOBAL_CHARTS_SAVER.save_model(self.model, X, y_)
        y_hat = self.model.predict(X)

        r2 = r2_score(y_, y_hat)
        MAPE = mean_absolute_percentage_error(y_, y_hat)
        self.logger.info(f"model r2: {r2}, MAPE: {MAPE}")


class KernelPCABO1(BO):
    """Dimensionality reduction using Principle Component Decomposition with kernel trick (Kernel-PCA)
    """

    def __init__(self, max_information_loss=0.2, kernel_fit_strategy: KernelFitStrategy = KernelFitStrategy.AUTO, kernel_config=None, NN: int = None, **kwargs):
        super().__init__(**kwargs)
        global GLOBAL_CHARTS_SAVER
        GLOBAL_CHARTS_SAVER = MyChartSaver('Kernel-PCABO-1', 'kernel-PCABO', self._search_space.bounds, self.obj_fun)
        if self.model is not None:
            self.logger.warning(
                "The surrogate model will be created automatically by PCA-BO. "
                "The input argument `model` will be ignored"
            )
        assert isinstance(self._search_space, RealSpace)
        self.__search_space = deepcopy(self._search_space)  # the original search space
        self._pca = KernelTransform(dimensions=self.search_space.dim, minimize=self.minimize, X_initial_space=[], epsilon=max_information_loss, kernel_fit_strategy=kernel_fit_strategy, kernel_config=kernel_config, NN=NN)
        self.acq_opt_time = 0
        self.mode_fit_time = 0

    @staticmethod
    def my_acquisition_function(x, lower_dimensional_acquisition_function, kpca):
        eprintf('acq x =', x)
        eprintf('acq lower dim x =', kpca.transform(x))
        return lower_dimensional_acquisition_function(kpca.transform(x)[0])

    def _create_acquisition(self, fun=None, par=None, return_dx=False, **kwargs) -> Callable:
        fun = fun if fun is not None else self._acquisition_fun
        par = copy(self._acquisition_par)
        par.update({"model": self.model, "minimize": self.minimize})
        criterion = getattr(AcquisitionFunction, fun)(**par)
        lower_dim_acq_function = partial_argument(
            func=functools.partial(criterion, return_dx=return_dx),
            var_name=self.lower_dim_search_space.var_name,
            fixed=None,
            reduce_output=return_dx,
        )
        return functools.partial(
            KernelPCABO1.my_acquisition_function,
            lower_dimensional_acquisition_function=lower_dim_acq_function,
            kpca=self._pca,
        )

    def pre_eval_check(self, X: List) -> List:
        """Note that we do not check against duplicated point in PCA-BO since those points are
        sampled in the reduced search space. Please check if this is intended
        """
        if isinstance(X, np.ndarray):
            X = X.tolist()
        return X

    @property
    def xopt(self):
        if not hasattr(self, "data"):
            return None
        fopt = self._get_best(self.data.fitness)
        self._xopt = self.data[np.where(self.data.fitness == fopt)[0][0]]
        return self._xopt

    def ask(self, n_point: int = None) -> List[List[float]]:
        return super().ask(n_point)

    @staticmethod
    def _compute_bounds(kpca: MyKernelPCA, search_space: SearchSpace) -> List[float]:
        if kpca.kernel_config['kernel_name'] == 'rbf':
            eprintf("The bounds are", search_space.bounds)
            c = [(lb + ub)/2 for lb,ub in search_space.bounds]
            corner = [ub for _,ub in search_space.bounds]
            eprintf("Mean in the initial space", c)
            half_diagonal_sqr = sum((ub - mean_i)**2 for ((_,ub), mean_i) in zip(search_space.bounds, c))
            eprintf("Squared half diagonal in the initial space is", half_diagonal_sqr)
            max_features_space_distance_from_mean = 2. - 2. * kpca.kernel(corner, c)
            eprintf(f'max feature space distance from the mean is {max_features_space_distance_from_mean}')
            C = kpca.transform([c])[0]
            eprintf(f'mean in the feature space is {C}')
            return [(_ - max_features_space_distance_from_mean, _ + max_features_space_distance_from_mean) for _ in C]
        elif kpca.kernel_config['kernel_name'] == 'polynomial':
            # TODO implement for the case when the bound box is not centred 
            c = [(lb + ub)/2 for lb,ub in search_space.bounds]
            corner = [ub for _,ub in search_space.bounds]
            origin = np.zeros(len(search_space.bounds))
            max_features_space_distance_from_origin = kpca.kernel(origin, origin) + kpca.kernel(corner, corner) - 2 * kpca.kernel(origin, corner)
            C = kpca.transform([origin])[0]
            return [(_ - max_features_space_distance_from_origin, _ + max_features_space_distance_from_origin) for _ in C]
        else:
            raise NotImplementedError

    def get_lower_space_dimensionality(self):
        return self._pca.k

    def get_extracted_information(self):
        return self._pca.extracted_information

    def tell(self, new_X, new_y):
        self.logger.info(f"observing {len(new_X)} points:")
        for i, x in enumerate(new_X):
            self.logger.info(f"#{i + 1} - fitness: {new_y[i]}, solution: {x}")

        index = np.arange(len(new_X))
        if hasattr(self, "data"):
            index += len(self.data)
        # convert `new_X` to a `Solution` object
        new_X = self._to_geno(new_X, index=index, n_eval=1, fitness=new_y)
        self.iter_count += 1
        self.eval_count += len(new_X)
        GLOBAL_CHARTS_SAVER.set_iter_number(self.iter_count)

        new_X = self.post_eval_check(new_X)  # remove NaN's
        self.data = self.data + new_X if hasattr(self, "data") else new_X
        # re-fit the PCA transformation
        self._pca.set_initial_space_points(self.data)
        X = self._pca.fit_transform(np.array(self.data), self.data.fitness)
        eprintf("Feature space points", X)
        self.update_model(X, self.data.fitness)
        GLOBAL_CHARTS_SAVER.save(self.iter_count, self.data)
        self.logger.info(f"xopt/fopt:\n{self.xopt}\n")

    def update_model(self, X: np.ndarray, y: np.ndarray):
        # NOTE: the GPR model will be created since the effective search space (the reduced space
        # is dynamic)
        bounds = np.asarray(self._compute_bounds(self._pca, self.__search_space))
        self.lower_dim_search_space = RealSpace(bounds)
        dim = self.lower_dim_search_space.dim
        self.model = GaussianProcess(
            mean=trend.constant_trend(dim),
            corr="matern",
            thetaL=1e-3 * (bounds[:, 1] - bounds[:, 0]),
            thetaU=1e3 * (bounds[:, 1] - bounds[:, 0]),
            nugget=1e-6,
            noise_estim=False,
            optimizer="BFGS",
            wait_iter=3,
            random_start=max(10, dim),
            likelihood="concentrated",
            eval_budget=100 * dim,
        )

        _std = np.std(y)
        y_ = y if np.isclose(_std, 0) else (y - np.mean(y)) / _std

        self.fmin, self.fmax = np.min(y_), np.max(y_)
        self.frange = self.fmax - self.fmin

        self.model.fit(X, y_)
        GLOBAL_CHARTS_SAVER.save_model(self.model, X, y_)
        y_hat = self.model.predict(X, batch_size=None)

        r2 = r2_score(y_, y_hat)
        MAPE = mean_absolute_percentage_error(y_, y_hat)
        self.logger.info(f"model r2: {r2}, MAPE: {MAPE}")


class ConditionalBO(ParallelBO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_optimizer(**kwargs)
        self._ucb = np.zeros(self.n_subspace)
        self._to_pheno = lambda x: x.to_dict()
        self._to_geno = lambda x, **kwargs: Solution.from_dict(x, **kwargs)
        self._bo_idx: List[int] = list()

    def _create_optimizer(self, **kwargs):
        _kwargs = kwargs.copy()
        _kwargs.pop("DoE_size", None)
        _kwargs.pop("n_point", None)
        _kwargs.pop("search_space", None)
        _kwargs.pop("eval_type", None)
        _kwargs.pop("model", None)
        self.subspaces = self.search_space.get_unconditional_subspace()
        self._bo = [
            BO(
                search_space=cs,
                DoE_size=1,
                n_point=1,
                eval_type="dict",
                model=RandomForest(levels=cs.levels),
                **_kwargs,
            )
            for _, cs in self.subspaces
        ]
        self.n_subspace = len(self.subspaces)
        self._init_gen = iter(range(self.n_subspace))
        self._fixed_vars = [d for d, _ in self.subspaces]

    def select_subspace(self, n_point: int) -> List[int]:
        if n_point == 0:
            return []
        return np.random.choice(self.n_subspace, n_point).tolist()

    @timeit
    def ask(self, n_point: int = None, fixed: Dict[str, Union[float, int, str]] = None) -> List[dict]:
        n_point = self.n_point if n_point is None else n_point
        idx = []
        # initial DoE
        for _ in range(n_point):
            try:
                idx.append(next(self._init_gen))
            except StopIteration:
                break

        # select subspaces/BOs
        idx += self.select_subspace(n_point=n_point - len(idx))
        self._bo_idx = idx
        # calling `ask` methods from BO's from each subspace
        X = [self._bo[i].ask()[0] for i in idx]
        # fill in the value for conditioning and irrelative variables (None)
        for i, k in enumerate(idx):
            X[i].update(self._fixed_vars[k])
            X[i].update({k: None for k in set(self.var_names) - set(X[i].keys())})
            self.logger.info(f"#{i + 1} - {X[i]}")
        return X

    @timeit
    def tell(
        self,
        X: List[Union[list, dict]],
        func_vals: List[Union[float, list]],
        warm_start: bool = False,
        **kwargs,
    ):
        """Tell the BO about the function values of proposed candidate solutions

        Parameters
        ----------
        X : List of Lists or Solution
            The candidate solutions which are usually proposed by the `self.ask` function
        func_vals : List/np.ndarray of reals
            The corresponding function values
        """
        assert len(self._bo_idx) == len(X)
        # call `tell` method of BOs in each subspace
        for i, k in enumerate(self._bo_idx):
            x = {k: v for k, v in X[i].items() if v}
            for key, _ in self._fixed_vars[k].items():
                x.pop(key)
            self._bo[k].tell([x], [func_vals[i]], **kwargs)

        X = self.post_eval_check(self._to_geno(X, fitness=func_vals))
        self.data = self.data + X if hasattr(self, "data") else X
        self.eval_count += len(X)

        xopt = self.xopt
        self.logger.info(f"fopt: {xopt.fitness}")
        self.logger.info(f"xopt: {self._to_pheno(xopt)}\n")

        if not warm_start:
            self.iter_count += 1
            self.hist_f.append(xopt.fitness)


class KernelPCABO(BO):
    """Dimensionality reduction using Principle Component Decomposition with kernel trick (Kernel-PCA)
    """

    def __init__(self, max_information_loss=0.2, kernel_fit_strategy: KernelFitStrategy = KernelFitStrategy.AUTO, kernel_config=None, NN: int = None, **kwargs):
        super().__init__(**kwargs)
        global GLOBAL_CHARTS_SAVER
        GLOBAL_CHARTS_SAVER = MyChartSaver('Kernel-PCABO-1', 'kernel-PCABO', self._search_space.bounds, self.obj_fun)
        if self.model is not None:
            self.logger.warning(
                "The surrogate model will be created automatically by PCA-BO. "
                "The input argument `model` will be ignored"
            )
        assert isinstance(self._search_space, RealSpace)
        self.__search_space = deepcopy(self._search_space)  # the original search space
        self._pca = KernelTransform(dimensions=self.search_space.dim, minimize=self.minimize, X_initial_space=[], epsilon=max_information_loss, kernel_fit_strategy=kernel_fit_strategy, kernel_config=kernel_config, NN=NN)
        self._pca.enable_inverse_transform(self.__search_space.bounds)
        self.out_solutions = 0
        self.ordered_container = KernelPCABO.MyOrderedContainer(self.minimize)
        self.ratio_of_best_for_kernel_refit = 0.2
        self.acq_opt_time = 0
        self.mode_fit_time = 0
        self.archive = KernelPCABO.Archive(self.dim, 3 * self.dim)

    @staticmethod
    def _compute_bounds(kpca: MyKernelPCA, search_space: SearchSpace) -> List[float]:
        if kpca.kernel_config['kernel_name'] == 'rbf':
            eprintf("The bounds are", search_space.bounds)
            c = [(lb + ub)/2 for lb,ub in search_space.bounds]
            corner = [ub for _,ub in search_space.bounds]
            eprintf("Mean in the initial space", c)
            half_diagonal_sqr = sum((ub - mean_i)**2 for ((_,ub), mean_i) in zip(search_space.bounds, c))
            eprintf("Squared half diagonal in the initial space is", half_diagonal_sqr)
            max_features_space_distance_from_mean = 2. - 2. * kpca.kernel(corner, c)
            eprintf(f'max feature space distance from the mean is {max_features_space_distance_from_mean}')
            C = kpca.transform([c])[0]
            eprintf(f'mean in the feature space is {C}')
            return [(_ - max_features_space_distance_from_mean, _ + max_features_space_distance_from_mean) for _ in C]
        elif kpca.kernel_config['kernel_name'] == 'polynomial':
            # TODO implement for the case when the bound box is not centred 
            c = [(lb + ub)/2 for lb,ub in search_space.bounds]
            corner = [ub for _,ub in search_space.bounds]
            origin = np.zeros(len(search_space.bounds))
            max_features_space_distance_from_origin = kpca.kernel(origin, origin) + kpca.kernel(corner, corner) - 2 * kpca.kernel(origin, corner)
            C = kpca.transform([origin])[0]
            return [(_ - max_features_space_distance_from_origin, _ + max_features_space_distance_from_origin) for _ in C]
        else:
            raise NotImplementedError

    def _create_acquisition(self, fun=None, par=None, return_dx=False, **kwargs) -> Callable:
        acquisition_func = super()._create_acquisition(
            fun=fun, par={} if par is None else par, return_dx=return_dx, **kwargs
        )
        # TODO: make this more general for other acquisition functions
        # wrap the penalized acquisition function for handling the box constraints
        # return functools.partial(
            # penalized_acquisition,
            # acquisition_func=acquisition_func,
            # bounds=self.__search_space.bounds,  # hyperbox in the original space
            # pca=self._pca,
            # return_dx=return_dx,
        # )
        return acquisition_func

    def pre_eval_check(self, X: List) -> List:
        """Note that we do not check against duplicated point in PCA-BO since those points are
        sampled in the reduced search space. Please check if this is intended
        """
        if isinstance(X, np.ndarray):
            X = X.tolist()
        return X

    @property
    def xopt(self):
        if not hasattr(self, "data"):
            return None
        fopt = self._get_best(self.data.fitness)
        self._xopt = self.data[np.where(self.data.fitness == fopt)[0][0]]
        return self._xopt

    class MyAcqOptimizationListener(OptimizationListener):
        def __init__(self):
            self.fopts = []
            self.xopts = []

        def on_optimum_found(self, fopt, xopt):
            self.fopts.append(fopt)
            self.xopts.append(xopt)

    def ask(self, n_point: int = None) -> List[List[float]]:
        eprintf("Beginning of acq optimization")
        listener = KernelPCABO.MyAcqOptimizationListener()
        start = time.time()
        X = super().ask(n_point, listener=listener)
        self.acq_opt_time = time.time() - start
        if len(X) > 1:
            return X
        inds = np.argsort(listener.fopts)[::-1]
        first_point = None
        bounds = self.__search_space.bounds
        for point_number, ind in enumerate(inds):
            new_point = self._pca.inverse_transform(listener.xopts[ind])[0]
            if point_number == 0:
                first_point = new_point
            is_out = False
            for i in range(len(bounds)):
                if new_point[i] < bounds[i][0]:
                    new_point[i] = bounds[i][0]
                    is_out = True
                if new_point[i] > bounds[i][1]:
                    new_point[i] = bounds[i][1]
                    is_out = True
            if not is_out:
                return [new_point]
        self.out_solutions += 1
        return [first_point]

    def _run_experiment(self, bounds):
        eprintf('==================== Experiment =========================')
        eprintf(bounds)
        sampled_points = self.__search_space._sample(1000)
        for p in sampled_points:
            y = self._pca.transform([p])[0]
            is_inside = sum(1 for i in range(len(y)) if bounds[i][0] <= y[i] <= bounds[i][1])==len(y)
            eprintf(p, '->', y, str(is_inside))
            if not is_inside:
                sys.exit(0)
        eprintf('==================== End =========================')
        sys.exit(0)

    def get_lower_space_dimensionality(self):
        return self._pca.k

    def get_extracted_information(self):
        return self._pca.extracted_information

    class MyOrderedContainer:
        def __init__(self, minimize=True):
            self.data = []
            self.minimize = minimize

        def __len__(self):
            return len(self.data)

        def add_element(self, element):
            if self.minimize:
                bisect.insort_right(self.data, element)
            else:
                bisect.insort_left(self.data, element)

        def kth(self, k):
            if not self.minimize:
                return self.data(len(self.data) - k)
            return self.data[k]

        def find_pos(self, element):
            if not self.minimize:
                return bisect.bisect_left(self.data, element)
            return bisect.bisect_right(self.data, element)


    def __is_promising_y(self, y):
        pos = self.ordered_container.find_pos(y)
        n = len(self.ordered_container)
        return pos / n <= self.ratio_of_best_for_kernel_refit if n != 0 else True


    class Archive:
        def __init__(self, old_threshold, bad_threshold, minimize=True):
            self.old_threshold = old_threshold
            self.bad_threshold = bad_threshold
            self.points = []
            self.__minimize = minimize
            if not minimize:
                raise NotImplementedError

        class Point:
            def __init__(self, coordinates, value):
                self.coordinates = coordinates
                self.obj_value = value
                self.old_count = 0

        def __update_after_insert(self, pos_insert):
            if pos_insert >= self.bad_threshold:
                return
            to_delete = []
            for i in range(pos_insert):
                self.points[i].old_count += 1
                if self.points[i].old_count >= self.old_threshold:
                    to_delete.append(i)
            for i in range(pos_insert, min(len(self.points), self.bad_threshold)):
                self.points[i].old_count = 0
            for ind in to_delete[::-1]:
                print(f'good point {ind} is thrown away')
                self.points.pop(ind)
            if len(to_delete)>0:
                for i in range(min(len(self.points), self.bad_threshold)):
                    self.points[i].old_count = 0
            print('point is good')
            [print(p.old_count, p.obj_value) for p in self.points]

        def add_point(self, coordinates, value):
            to_delete_positions = []
            for i in range(min(len(self.points), self.bad_threshold)):
                if self.points[i].obj_value > value:
                    self.points.insert(i, KernelPCABO.Archive.Point(coordinates, value))
                    self.__update_after_insert(i)
                    return

        def get_points_for_manifold_learning(self):
            return np.array([p.coordinates for p in self.points]), np.array([p.obj_value for p in self.points])

        def add_doe(self, coordinates, values):
            ids = np.argsort(values)
            self.points = [KernelPCABO.Archive.Point([], 0)] * len(values)
            pos = 0
            for ind in ids:
                self.points[pos] = KernelPCABO.Archive.Point(coordinates[ind], values[ind])
                pos += 1


    def tell(self, new_X, new_y):
        self.logger.info(f"observing {len(new_X)} points:")
        for i, x in enumerate(new_X):
            self.logger.info(f"#{i + 1} - fitness: {new_y[i]}, solution: {x}")

        index = np.arange(len(new_X))
        if hasattr(self, "data"):
            index += len(self.data)
        # convert `new_X` to a `Solution` object
        new_X = self._to_geno(new_X, index=index, n_eval=1, fitness=new_y)
        self.iter_count += 1
        self.eval_count += len(new_X)
        GLOBAL_CHARTS_SAVER.set_iter_number(self.iter_count)

        new_X = self.post_eval_check(new_X)  # remove NaN's
        if len(new_X) > 1:
            self.archive.add_doe(new_X, new_y)
        else:
            self.archive.add_point(new_X[0], new_y[0])
        self.data = self.data + new_X if hasattr(self, "data") else new_X
        for y in new_y:
            if self.__is_promising_y(y):
                self._pca.set_kernel_refit_needed(True)
            self.ordered_container.add_element(y)
        # re-fit the PCA transformation
        good_points, good_points_values = self.archive.get_points_for_manifold_learning()
        self._pca.set_initial_space_points(good_points)
        self._pca.fit(good_points, good_points_values)
        X = self._pca.transform(self.data)
        eprintf("Feature space points", X)
        bounds = self._compute_bounds(self._pca, self.__search_space)
        eprintf("Bounds in the feature space", bounds)
        # self._run_experiment(bounds)
        # re-set the search space object for the reduced (feature) space
        self._search_space = RealSpace(bounds)
        GLOBAL_CHARTS_SAVER.save_with_manifold(self.iter_count, self.data, X, bounds[0][0], bounds[0][1], self._pca)
        # update the surrogate model
        self.update_model(X, self.data.fitness)
        self.logger.info(f"xopt/fopt:\n{self.xopt}\n")

    def update_model(self, X: np.ndarray, y: np.ndarray):
        # NOTE: the GPR model will be created since the effective search space (the reduced space
        # is dynamic)
        dim = self._search_space.dim
        bounds = np.asarray(self._search_space.bounds)
        self._model = self.create_default_model(self._search_space, self.my_seed)
        self.model = clone(self._model)

        _std = np.std(y)
        y_ = y

        self.fmin, self.fmax = np.min(y_), np.max(y_)
        self.frange = self.fmax - self.fmin

        start = time.time()
        self.model.fit(X, y_)
        self.mode_fit_time = time.time() - start
        GLOBAL_CHARTS_SAVER.save_model(self.model, X, y_)
        y_hat = self.model.predict(X)

        r2 = r2_score(y_, y_hat)
        MAPE = mean_absolute_percentage_error(y_, y_hat)
        self.logger.info(f"model r2: {r2}, MAPE: {MAPE}")


class ConditionalBO(ParallelBO):
    def __init__(self, **kwargs):
        super().__init__(model=RandomForest(levels={}), **kwargs)
        self._create_optimizer(**kwargs)
        self._ucb = np.zeros(self.n_subspace)
        self._to_pheno = lambda x: x.to_dict()
        self._to_geno = lambda x, **kwargs: Solution.from_dict(x, **kwargs)
        self._bo_idx: List[int] = list()

    def _create_optimizer(self, **kwargs):
        _kwargs = kwargs.copy()
        _kwargs.pop("DoE_size", None)
        _kwargs.pop("n_point", None)
        _kwargs.pop("search_space", None)
        _kwargs.pop("eval_type", None)
        _kwargs.pop("model", None)
        self.subspaces = self.search_space.get_unconditional_subspace()
        self._bo = [
            BO(
                search_space=cs,
                DoE_size=1,
                n_point=1,
                eval_type="dict",
                model=RandomForest(levels=cs.levels),
                **_kwargs,
            )
            for _, cs in self.subspaces
        ]
        self.n_subspace = len(self.subspaces)
        self._init_gen = iter(range(self.n_subspace))
        self._fixed_vars = [d for d, _ in self.subspaces]

    def select_subspace(self, n_point: int) -> List[int]:
        if n_point == 0:
            return []
        return np.random.choice(self.n_subspace, n_point).tolist()

    @timeit
    def ask(self, n_point: int = None, fixed: Dict[str, Union[float, int, str]] = None) -> List[dict]:
        n_point = self.n_point if n_point is None else n_point
        idx = []
        # initial DoE
        for _ in range(n_point):
            try:
                idx.append(next(self._init_gen))
            except StopIteration:
                break

        # select subspaces/BOs
        idx += self.select_subspace(n_point=n_point - len(idx))
        self._bo_idx = idx
        # calling `ask` methods from BO's from each subspace
        X = [self._bo[i].ask()[0] for i in idx]
        # fill in the value for conditioning and irrelative variables (None)
        for i, k in enumerate(idx):
            X[i].update(self._fixed_vars[k])
            X[i].update({k: None for k in set(self.var_names) - set(X[i].keys())})
            self.logger.info(f"#{i + 1} - {X[i]}")
        return X

    @timeit
    def tell(
        self,
        X: List[Union[list, dict]],
        func_vals: List[Union[float, list]],
        warm_start: bool = False,
        **kwargs,
    ):
        """Tell the BO about the function values of proposed candidate solutions

        Parameters
        ----------
        X : List of Lists or Solution
            The candidate solutions which are usually proposed by the `self.ask` function
        func_vals : List/np.ndarray of reals
            The corresponding function values
        """
        assert len(self._bo_idx) == len(X)
        # call `tell` method of BOs in each subspace
        for i, k in enumerate(self._bo_idx):
            x = {k: v for k, v in X[i].items() if v}
            for key, _ in self._fixed_vars[k].items():
                x.pop(key)
            self._bo[k].tell([x], [func_vals[i]], **kwargs)

        X = self.post_eval_check(self._to_geno(X, fitness=func_vals))
        self.data = self.data + X if hasattr(self, "data") else X
        self.eval_count += len(X)

        xopt = self.xopt
        self.logger.info(f"fopt: {xopt.fitness}")
        self.logger.info(f"xopt: {self._to_pheno(xopt)}\n")

        if not warm_start:
            self.iter_count += 1
            self.hist_f.append(xopt.fitness)


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
            _criterion = getattr(AcquisitionFunction, self._acquisition_fun_list[i])(model=kwargs["model"])
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
