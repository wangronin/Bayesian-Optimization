from __future__ import annotations

import functools
from copy import copy, deepcopy
from typing import Callable, Dict, List, Union

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from . import acquisition_fun as AcquisitionFunction
from .bayes_opt import BO, ParallelBO
from .search_space import RealSpace, SearchSpace
from .solution import Solution
from .surrogate import GaussianProcess, RandomForest
from .utils import timeit
from .kpca import MyKernelPCA

from .mylogging import *
import time


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


class KernelTransform(MyKernelPCA):
    def __init__(self, minimize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.minimize = minimize

    @staticmethod
    def _check_input(X):
        return np.atleast_2d(np.asarray(X, dtype=float))

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = self._check_input(X)
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> KernelTransform:
        X_scaled = self._weighting_scheme(X, y)
        self._fit_kernel_parameters(X_scaled)
        return super().fit(X_scaled)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._check_input(X)
        return super().transform(X)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        Y = self._check_input(Y)
        return super().inverse_transform(Y)

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

    def _fit_kernel_parameters(self, X):
        pass


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

    @staticmethod
    def _compute_bounds(pca: PCA, search_space: SearchSpace) -> List[float]:
        C = np.array([(l + u) / 2 for l, u in search_space.bounds])
        radius = np.sqrt(np.sum((np.array([l for l, _ in search_space.bounds]) - C) ** 2))
        C = C - pca.mean_ - pca.center
        C_ = C.dot(pca.components_.T)
        return [(_ - radius, _ + radius) for _ in C_]

    def _create_acquisition(self, fun=None, par=None, return_dx=False, **kwargs) -> Callable:
        acquisition_func = super()._create_acquisition(
            fun=fun, par={} if par is None else par, return_dx=return_dx, **kwargs
        )
        # TODO: make this more general for other acquisition functions
        # wrap the penalized acquisition function for handling the box constraints
        return functools.partial(
            penalized_acquisition,
            acquisition_func=acquisition_func,
            bounds=self.__search_space.bounds,  # hyperbox in the original space
            pca=self._pca,
            return_dx=return_dx,
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
        new_x1 = self._pca.inverse_transform(super().ask(n_point))
        new_x = new_x1[0]
        is_inside = sum(1 for i in range(len(new_x)) if self.__search_space.bounds[i][0]<= new_x[i] <= self.__search_space.bounds[i][1])==len(new_x)
        eprintf("Is inside?", is_inside)
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
        GLOBAL_CHARTS_SAVER.save_with_manifold(self.iter_count, self.data, X, self._pca)

        bounds = self._compute_bounds(self._pca, self.__search_space)
        # re-set the search space object for the reduced (feature) space
        self._search_space = RealSpace(bounds)
        # update the surrogate model
        self.update_model(X, self.data.fitness)
        self.logger.info(f"xopt/fopt:\n{self.xopt}\n")

    def update_model(self, X: np.ndarray, y: np.ndarray):
        # NOTE: the GPR model will be created since the effective search space (the reduced space
        # is dynamic)
        dim = self._search_space.dim
        self.model = GaussianProcess(domain=self._search_space, n_restarts_optimizer=dim)

        _std = np.std(y)
        y_ = y if np.isclose(_std, 0) else (y - np.mean(y)) / _std

        self.fmin, self.fmax = np.min(y_), np.max(y_)
        self.frange = self.fmax - self.fmin

        self.model.fit(X, y_)
        GLOBAL_CHARTS_SAVER.save_model(self.model, X, y_)
        y_hat = self.model.predict(X)

        r2 = r2_score(y_, y_hat)
        MAPE = mean_absolute_percentage_error(y_, y_hat)
        self.logger.info(f"model r2: {r2}, MAPE: {MAPE}")


class KernelPCABO(BO):
    """Dimensionality reduction using Principle Component Decomposition with kernel trick (Kernel-PCA)
    """

    def __init__(self, n_components: Union[float, int] = None, **kwargs):
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
        self._pca = KernelTransform(minimize=self.minimize, X_initial_space=[], kernel_name='rbf', epsilon=0.7, kernel_params_dict={'gamma': 0.0001, 'd': 2, 'c0': 0})

    @staticmethod
    def _compute_bounds(kpca: MyKernelPCA, search_space: SearchSpace) -> List[float]:
        if kpca.kernel_name == 'rbf':
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
        elif kpca.kernel_name == 'polynomial':
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
        return functools.partial(
            penalized_acquisition,
            acquisition_func=acquisition_func,
            bounds=self.__search_space.bounds,  # hyperbox in the original space
            pca=self._pca,
            return_dx=return_dx,
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
        eprintf("Beginning of acq optimization")
        return self._pca.inverse_transform(super().ask(n_point))

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
        self.model = GaussianProcess(domain=self._search_space, n_restarts_optimizer=dim)

        _std = np.std(y)
        y_ = y if np.isclose(_std, 0) else (y - np.mean(y)) / _std

        self.fmin, self.fmax = np.min(y_), np.max(y_)
        self.frange = self.fmax - self.fmin

        self.model.fit(X, y_)
        GLOBAL_CHARTS_SAVER.save_model(self.model, X, y_)
        y_hat = self.model.predict(X)

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
