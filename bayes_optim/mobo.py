import functools
from typing import Dict, List, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor

from .acquisition_optim import argmax_restart
from .bayes_opt import BO
from .extra.multi_objective import Hypervolume, is_non_dominated
from .extra.multi_objective.box_decompositions import NondominatedPartitioning
from .multi_objective import EHVI
from .utils import dynamic_penalty, partial_argument, timeit

__authors__ = ["Hao Wang"]


class BaseMOBO(BO):
    """Base class for multi-objective BO"""

    def __init__(self, n_obj: int = 2, minimize: Union[bool, List[bool]] = True, **kwargv):
        # NOTE: setting `minimize=True` to make parent's constructor work
        super().__init__(minimize=True, **kwargv)
        self._check_obj_fun(n_obj)
        self._check_minimize(minimize)

    def _check_minimize(self, minimize):
        self.minimize = minimize
        if isinstance(self.minimize, bool):
            self.minimize = [self.minimize] * self.n_obj
        elif hasattr(self.minimize, "__iter__"):
            assert len(self.minimize) == self.n_obj
        self.minimize = np.asarray(self.minimize)

    def _check_obj_fun(self, n_obj):
        """check the objective functions"""
        self.n_obj = n_obj
        if self.obj_fun is None:
            return
        assert hasattr(self.obj_fun, "__iter__")
        assert all([hasattr(f, "__call__") for f in self.obj_fun])
        if len(self.obj_fun) != n_obj:
            self.logger.warning(
                f"len(obj_fun) ({self.obj_fun}) != n_obj ({n_obj})."
                "Setting n_obj according to the former."
            )
            self.n_obj = len(self.obj_fun)
        assert self.n_obj > 1

    @property
    def xopt(self):
        """get the non-dominated subset of solutions"""
        if self.y is None:
            return None
        idx = is_non_dominated(Tensor(self.y)).detach().cpu().numpy()
        return self.data[idx, :]

    @property
    def ref_point(self):
        """reference point for computing the hypervolumne"""
        # NOTE: assuming maximization
        return np.min(self.y, axis=0) * 0.8

    @property
    def y(self):
        """The transformed objective value for the internal use"""
        if not hasattr(self, "data"):
            return None
        # normalize the objective values in case objective functions differ hugely in their ranges
        self._scaler = MinMaxScaler()
        y = self._scaler.fit_transform(self.data.fitness)
        # y = self.data.fitness
        # convert to maximization problem
        return y * (-1) ** self.minimize

    @timeit
    def evaluate(self, X) -> List[Tuple[float]]:
        """evaluation

        Returns
        -------
        List[Tuple[float]]
            of the shape [[f1(x1),...,fm(x1)], [f1(x2),...,fm(x2)],...,[f1(xn),...,fm(xn)]]
        """
        func_vals = []
        for f in self.obj_fun:
            if self.n_job > 1:  # or by ourselves..
                func_vals.append(Parallel(n_jobs=self.n_job)(delayed(f)(x) for x in X))
            else:  # or sequential execution
                func_vals.append([f(x) for x in X])

        return list(zip(*func_vals))

    @timeit
    def tell(
        self,
        X: List[Union[list, dict]],
        func_vals: List[List[float]],
        h_vals: List[List[float]] = None,
        g_vals: List[List[float]] = None,
        index: List[str] = None,
        warm_start: bool = False,
    ):
        """Tell the BO about the function values of proposed candidate solutions

        Parameters
        ----------
        X : List of Lists or Solution
            The candidate solutions which are usually proposed by the `self.ask` function
        func_vals : List/np.ndarray of float
            The corresponding function values
        """
        X = self._to_geno(X, index)
        self.logger.info(f"observing {len(X)} points:")

        # TODO: implement method to handle known, expensive constraints `h_vals` and `g_vals`
        # add extra columns h_vals, g_vals to the `Solution` object
        for i, _ in enumerate(X):
            X[i].n_eval += 1
            for k in range(self.n_obj):
                X[i].fitness[k] = func_vals[i][k]
                if not warm_start:
                    self.eval_count += 1
            _fitness = ", ".join([f"f{_ + 1}: {func_vals[i][_]}" for _ in range(self.n_obj)])
            self.logger.info(f"#{i + 1} - {_fitness}, solution: {self._to_pheno(X[i])}")

        X = self.post_eval_check(X)
        if self.data_file is not None:
            X.to_csv(self.data_file, header=False, append=True)
        self.data = self.data + X if hasattr(self, "data") else X

        # update the surrogate models
        self.update_model()
        # compute the hypervolume indicator value of the incumbent
        # TODO: HV computation is not exactly correct
        xopt = self.xopt
        pf = self._scaler.transform(self.xopt.fitness) * (-1) ** self.minimize
        hv = Hypervolume(ref_point=Tensor(self.ref_point))
        self.logger.info(f"efficient set/Pareto front (xopt/fopt):\n{xopt}")
        self.logger.info(f"hypervolume of normalized fitness: {hv.compute(Tensor(pf))}")

        if self.h is not None or self.g is not None:
            penalty = np.array(
                [dynamic_penalty(x, 1, self._h, self._g, minimize=False) for x in xopt]
            ).ravel()
            self.logger.info(f"... with corresponding penalty: {penalty}")

        if not warm_start:
            self.iter_count += 1
            self.hist_f.append(xopt.fitness)

    def update_model(self):
        # TODO: add model selection to the base class
        X, y = self.data, self.y
        self.model.fit(X, y)
        y_ = self.model.predict(X)
        r2 = r2_score(y, y_, multioutput="raw_values")
        MAPE = mean_absolute_percentage_error(y, y_, multioutput="raw_values")

        for i in range(self.n_obj):
            _r2 = r2[i] if not isinstance(r2, float) else r2
            _mape = MAPE[i] if not isinstance(MAPE, float) else MAPE
            self.logger.info(f"model of f{i + 1} r2: {_r2}, MAPE: {_mape}")


class MOBO(BaseMOBO):
    """EHVI with Kriging believer"""

    def __init__(self, *args, **kwargv):
        super().__init__(*args, **kwargv)
        if self._acquisition_fun != "EHVI":
            self._acquisition_fun = "EHVI"
            self.logger.warning(
                "MOBO only allows using `EHIV` acquisition function. Ignore user's argument."
            )

    def _create_acquisition(self, fixed: Dict = None, **kwargv):
        # TODO: implement the Kriging believer strategy
        partitioning = NondominatedPartitioning(ref_point=Tensor(self.ref_point), Y=Tensor(self.y))
        criterion = EHVI(
            model=self.model, ref_point=self.ref_point.tolist(), partitioning=partitioning
        )
        return partial_argument(
            functools.partial(criterion),
            self.search_space.var_name,
            fixed,
            reduce_output=False,
        )


class MOBO_qEHVI(BaseMOBO):
    """qEHVI (q-point Expected Hypervolume Improvement)"""

    def __init__(self, *args, **kwargv):
        super().__init__(*args, **kwargv)
        if self._acquisition_fun != "qEHVI":
            self._acquisition_fun = "qEHVI"
            self.logger.warning(
                "MOBO only allows using `qEHIV` acquisition function. Ignore user's argument."
            )

    # def _create_acquisition(
    #     self, fun: str = None, par: dict = None, return_dx: bool = False, fixed: Dict = None
    # ):
    #     ref_point = np.max(self.y, axis=0) * 1.3
    #     partitioning = NondominatedPartitioning(ref_point=Tensor(ref_point), Y=Tensor(self.y))
    #     return qEHVI(
    #         model=par["model"],
    #         ref_point=ref_point.tolist(),
    #         n_point=par["n_point"],
    #         partitioning=partitioning,
    #     )

    def _batch_arg_max_acquisition(
        self, n_point: int, return_dx: bool, fixed: Dict = None
    ) -> Tuple[List[list], List[float]]:
        """Set ``self._argmax_restart`` for optimizing the acquisition function"""
        fixed = {} if fixed is None else fixed
        mask = np.array([v in fixed.keys() for v in self._search_space.var_name])
        values = [fixed[k] for i, k in enumerate(self._search_space.var_name) if mask[i]]
        idx = np.nonzero(~mask)[0]
        _argmax_restart = functools.partial(
            argmax_restart,
            search_space=self._search_space[idx] * n_point,
            h=((self._h, np.repeat(mask, n_point), values * n_point) if self._h else None),
            g=(
                partial_argument(self._g, np.repeat(mask, n_point), values * n_point)
                if self._g
                else None
            ),
            eval_budget=self.AQ_max_FEs,
            n_restart=self.AQ_n_restart,
            wait_iter=self.AQ_wait_iter,
            optimizer=self._optimizer,
        )
        candidates, values = _argmax_restart(
            self._create_acquisition(par={"n_point": n_point, "model": self.model}, fixed=fixed),
            logger=self.logger,
        )
        candidates = [candidates[i * self.dim : (i + 1) * self.dim] for i in range(n_point)]
        return candidates, [values] * n_point
