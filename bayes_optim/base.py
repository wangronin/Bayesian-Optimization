import functools
import logging
import os
import sys
from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dill
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from . import acquisition_fun as AcquisitionFunction
from ._exception import AskEmptyError, FlatFitnessError, RecommendationUnavailableError
from .acquisition_optim import argmax_restart
from .acquisition_optim.option import (
    default_AQ_max_FEs,
    default_AQ_n_restart,
    default_AQ_wait_iter,
)
from .misc import LoggerFormatter
from .search_space import RealSpace, SearchSpace
from .solution import Solution
from .utils import (
    arg_to_int,
    dynamic_penalty,
    fillin_fixed_value,
    func_with_list_arg,
    partial_argument,
    timeit,
)

__authors__ = ["Hao Wang"]


class BaseBO(ABC):
    """Bayesian Optimization base class, which implements the Ask-Evaluate-Tell interface"""

    def __init__(
        self,
        search_space: SearchSpace,
        obj_fun: Optional[Callable] = None,
        parallel_obj_fun: Optional[Callable] = None,
        eq_fun: Optional[Callable] = None,
        ineq_fun: Optional[Callable] = None,
        model: Optional[Any] = None,
        eval_type: str = "list",
        DoE_size: Optional[int] = None,
        warm_data: Tuple = (),
        n_point: int = 1,
        acquisition_fun: str = "EI",
        acquisition_par: dict = {},
        acquisition_optimization: dict = {},
        ftarget: Optional[float] = None,
        max_FEs: Optional[int] = None,
        minimize: bool = True,
        n_job: int = 1,
        data_file: Optional[str] = None,
        verbose: bool = False,
        random_seed: Optional[int] = None,
        logger: Optional[str] = None,
        instance_name: Optional[str] = None,
        **kwargs,
    ):
        """The base class for Bayesian Optimization

        Parameters
        ----------
        search_space : SearchSpace
            The search space, an instance of `SearchSpace` class.
        obj_fun : Callable
            The objective function to optimize.
        parallel_obj_fun : Callable, optional
            The objective function that takes multiple solutions simultaneously and
            implement the parallelization by itself, by default None.
        eq_fun : Callable, optional
            The equality constraints, whose return value should have the same size as the
            number of equality constraints, by default None.
        ineq_fun : Callable, optional
            The inequality constraints, whose return value should have the same size as
            the number of inequality constraints, by default None.
        model : Any, optional
            The surrogate mode, which will be automatically created if not passed in,
            by default None.
        eval_type : str, optional
            The type of input argument allowed by `obj_func` or `parallel_obj_fun`:
            it could be either 'list' or 'dict', by default 'list'.
        DoE_size : int, optional
            The size of inital Design of Experiment (DoE), by default None.
        warm_data: Tuple, optional
            The warm-starting data in a pair of design points and its objective values
            `(X, y)`, where `X` should be a list of points and has the same length with `y`.
            When provided, the initial sampling (DoE) operation is skipped.
        n_point : int, optional
            The number of candidate solutions proposed using infill-criteria, by default 1
        acquisition_fun : str, optional
            The acquisition function, by default 'EI'
        acquisition_par : dict, optional
            Extra parameters to the acquisition function, by default {}
        acquisition_optimization : dict, optional
            Additional parameters controlling the acquisition optimization, by default {}
        ftarget : float, optional
            The target value to hit, by default None
        max_FEs : int, optional
            The maximal number of evaluations, by default None
        minimize : bool, optional
            To minimize or maximize, by default True
        n_job : int, optional
            The number of allowable jobs for parallelizing the function evaluation
            (if `parallel_obj_fun` is not specified) and Only Effective when n_point > 1,
            by default 1
        data_file : str, optional
            The name of the file to store extra historical information during the run,
            by default None
        verbose : bool, optional
            The verbosity, by default False
        random_seed : int, optional
            The seed for pseudo-random number generators, by default None
        logger : str, optional
            Name of the logger file, by default None, which turns off the logging behaviour
        """
        self.obj_fun = obj_fun
        self.parallel_obj_fun = parallel_obj_fun
        self.h = eq_fun
        self.g = ineq_fun
        self.n_obj: int = 1
        self.n_job: int = max(1, int(n_job))
        self.n_point: int = max(1, int(n_point))
        self.ftarget = ftarget
        self.minimize = minimize
        self.verbose = verbose
        self.data_file = data_file
        self.max_FEs = int(max_FEs) if max_FEs else np.inf
        self.instance_name = instance_name
        self.metric_meta = None

        self.search_space = search_space
        self.DoE_size = DoE_size

        self.acquisition_fun = acquisition_fun
        self._acquisition_par = acquisition_par
        self._acquisition_callbacks = []  # the callback functions executed after
        # every call of `arg_max_acquisition`
        self.model = model
        self.logger = logger
        self.random_seed = random_seed

        self._eval_type = eval_type
        self._set_internal_optimization(**acquisition_optimization)
        self._get_best = np.min if self.minimize else np.max
        self._init_flatfitness_trial = 2
        self._set_aux_vars()
        self.warm_data = warm_data

    @property
    def acquisition_fun(self):
        return self._acquisition_fun

    @acquisition_fun.setter
    def acquisition_fun(self, fun):
        if isinstance(fun, str):
            self._acquisition_fun = fun
        else:
            assert hasattr(fun, "__call__")
        self._acquisition_fun = fun

    @property
    def DoE_size(self):
        return self._DoE_size

    @DoE_size.setter
    def DoE_size(self, DoE_size):
        if DoE_size:
            if isinstance(DoE_size, str):
                self._DoE_size = int(eval(DoE_size))
            elif isinstance(DoE_size, (int, float)):
                self._DoE_size = int(DoE_size)
            else:
                raise ValueError
        else:
            self._DoE_size = int(self.dim * 5)

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed):
        if seed:
            self._random_seed = int(seed)
            if self._random_seed:
                np.random.seed(self._random_seed)

    @property
    def search_space(self):
        return self._search_space

    @search_space.setter
    def search_space(self, search_space):
        self._search_space = search_space
        self.dim = len(self._search_space)
        self.var_names = self._search_space.var_name
        self.r_index = self._search_space.real_id  # indices of continuous variable
        self.i_index = self._search_space.integer_id  # indices of integer variable
        # indices of categorical variable
        self.d_index = self._search_space.categorical_id

        self.param_type = self._search_space.var_type
        self.N_r = len(self.r_index)
        self.N_i = len(self.i_index)
        self.N_d = len(self.d_index)

    @property
    def warm_data(self):
        return self._warm_data

    @warm_data.setter
    def warm_data(self, data):
        assert self.iter_count == 0  # warm data should only be provided in the begining
        if data is None or len(data) == 0:
            self._warm_data = None
        else:
            X, y = data
            assert len(X) == len(y)

            if isinstance(X, Solution):
                assert X.var_name == self.var_names
            else:
                X = self._to_geno(X)

            X.fitness = y
            X.n_eval = 1
            assert all([isinstance(_, float) for _ in X[:, self.r_index].ravel()])
            assert all([isinstance(_, int) for _ in X[:, self.i_index].ravel()])
            assert all([isinstance(_, str) for _ in X[:, self.d_index].ravel()])
            self._warm_data = X
            self.tell(X, y, warm_start=True)

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        if isinstance(logger, logging.Logger):
            self._logger = logger
            self._logger.propagate = False
            return

        # NOTE: logging.getLogger create new instance based on `name`
        # no new instance will be created if the same name is provided
        name = self.instance_name if self.instance_name else str(id(self))
        self._logger = logging.getLogger(f"{self.__class__.__name__} ({name})")
        self._logger.setLevel(logging.DEBUG)
        fmt = LoggerFormatter()

        # create console handler and set level to the vebosity
        SH = list(filter(lambda h: isinstance(h, logging.StreamHandler), self._logger.handlers))
        if self.verbose and len(SH) == 0:
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.INFO)
            sh.setFormatter(fmt)
            self._logger.addHandler(sh)

        # create file handler and set level to debug
        # TODO: perhaps also according to the verbosity?
        # TODOL perhaps create a logger class
        FH = list(filter(lambda h: isinstance(h, logging.FileHandler), self._logger.handlers))
        if logger is not None and len(FH) == 0:
            fh = logging.FileHandler(logger)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

        if hasattr(self, "logger"):
            self._logger.propagate = False

    def _set_aux_vars(self):
        self.iter_count = 0
        self.eval_count = 0
        self.stop_dict = {}
        self.hist_f = []

        if self._eval_type == "list":
            self._to_pheno = lambda x: copy(x.tolist())
            self._to_geno = lambda x, index=None, n_eval=1, fitness=None: Solution(
                x,
                var_name=self.var_names,
                n_obj=self.n_obj,
                index=index,
                n_eval=n_eval,
                fitness=fitness,
            )
        elif self._eval_type == "dict":
            self._to_pheno = lambda x: x.to_dict().copy()
            self._to_geno = lambda x, index=None: Solution.from_dict(
                x, index=index, n_obj=self.n_obj
            )

    def _set_internal_optimization(self, **kwargs):
        # TODO: turn this into an Option Class
        if "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]
        else:
            if isinstance(self.search_space, RealSpace):
                if hasattr(self.model, "gradient") and self.n_obj == 1:
                    self._optimizer = "BFGS"
                else:
                    self._optimizer = "OnePlusOne_Cholesky_CMA"
            else:
                self._optimizer = "MIES"

        if self._optimizer == "BFGS" and (self.h or self.g):
            self._optimizer = "OnePlusOne_Cholesky_CMA"

        # NOTE: `AQ` -> acquisition
        if "max_FEs" in kwargs:
            self.AQ_max_FEs = arg_to_int(kwargs["max_FEs"])
        else:
            self.AQ_max_FEs = default_AQ_max_FEs[self._optimizer](self.dim)

        self.AQ_n_restart = (
            default_AQ_n_restart(self.dim)
            if "n_restart" not in kwargs
            else arg_to_int(kwargs["n_restart"])
        )
        self.AQ_wait_iter = (
            default_AQ_wait_iter if "wait_iter" not in kwargs else arg_to_int(kwargs["wait_iter"])
        )

        # NOTE: `_h` and `_g` are wrappers of `h` and `g`, which always take lists as input
        self._h, self._g = self.h, self.g
        if self._h is not None:
            self._h = func_with_list_arg(self._h, self._eval_type, self.search_space.var_name)
        if self._g is not None:
            self._g = func_with_list_arg(self._g, self._eval_type, self.search_space.var_name)

    def __set_argmax(self, fixed: Dict = None):
        """Set ``self._argmax_restart`` for optimizing the acquisition function"""
        fixed = {} if fixed is None else fixed
        self._argmax_restart = functools.partial(
            argmax_restart,
            search_space=self.search_space.filter(fixed.keys(), invert=True),
            h=partial_argument(self._h, self.search_space.var_name, fixed) if self._h else None,
            g=partial_argument(self._g, self.search_space.var_name, fixed) if self._g else None,
            eval_budget=self.AQ_max_FEs,
            n_restart=self.AQ_n_restart,
            wait_iter=self.AQ_wait_iter,
            optimizer=self._optimizer,
        )

    def _check_params(self):
        # TODO: add more parameter check-ups
        if np.isinf(self.max_FEs):
            raise ValueError("max_FEs cannot be infinite")

        assert hasattr(AcquisitionFunction, self._acquisition_fun)

    def _compare(self, f1: float, f2: float) -> bool:
        """Test if objecctive value f1 is better than f2"""
        return f1 < f2 if self.minimize else f2 > f1

    @property
    def xopt(self):
        if not hasattr(self, "data"):
            return None
        fopt = self._get_best(self.data.fitness)
        self._xopt = self.data[np.where(self.data.fitness == fopt)[0][0]]
        return self._xopt

    def run(self) -> Tuple[List[Solution], dict]:
        while not self.check_stop():
            self.step()
        return self._to_pheno(self.xopt), self.xopt.fitness, self.stop_dict

    def step(self):
        self.logger.info(f"iteration {self.iter_count} starts...")
        X = self.ask()
        # TODO: add exception handling for evaluating the objective function
        func_vals = self.evaluate(X)
        self.tell(X, func_vals)

    @timeit
    def ask(
        self, n_point: int = None, fixed: Dict[str, Union[float, int, str]] = None
    ) -> Union[List[list], List[dict]]:
        """suggest a list of candidate solutions

        Parameters
        ----------
        n_point : int, optional
            the number of candidates to request, by default None
        fixed : Dict[str, Union[float, int, str]], optional
            a dictionary specifies the decision variables fixed and the value to which those
            are fixed, by default None

        Returns
        -------
        Union[List[list], List[dict]]
            the suggested candidates
        """
        if self.model is not None and self.model.is_fitted:
            n_point = self.n_point if n_point is None else n_point
            msg = f"asking {n_point} points:"
            X = self.arg_max_acquisition(n_point=n_point, fixed=fixed)
            X = self.pre_eval_check(X)  # validate the new candidate solutions
            if len(X) < n_point:
                self.logger.warning(
                    f"iteration {self.iter_count}: duplicated solution found "
                    "by optimization! New points is taken from random design."
                )
                N = n_point - len(X)
                # take random samples if the acquisition optimization failed
                X += self.create_DoE(N, fixed=fixed)
        else:  # take the initial DoE
            n_point = self._DoE_size if n_point is None else n_point
            msg = f"asking {n_point} points (using DoE):"
            X = self.create_DoE(n_point, fixed=fixed)

        if len(X) == 0:
            raise AskEmptyError()

        index = np.arange(len(X))
        if hasattr(self, "data"):
            index += len(self.data)

        X = Solution(X, index=index, var_name=self._search_space.var_name)
        self.logger.info(msg)
        for i, _ in enumerate(X):
            self.logger.info(f"#{i + 1} - {self._to_pheno(X[i])}")

        return self._to_pheno(X)

    @timeit
    def tell(
        self,
        X: List[Union[list, dict]],
        func_vals: List[Union[float, list]],
        h_vals: List[Union[float, list]] = None,
        g_vals: List[Union[float, list]] = None,
        index: List[str] = None,
        warm_start: bool = False,
    ):
        """Tell the BO about the function values of proposed candidate solutions

        Parameters
        ----------
        X : List of Lists or Solution
            The candidate solutions which are usually proposed by the `self.ask` function
        func_vals : List/np.ndarray of reals
            The corresponding function values
        """
        # TODO: implement method to handle known, expensive constraints `h_vals` and `g_vals`
        X = self._to_geno(X, index)
        self.logger.info(f"observing {len(X)} points:")
        for i, _ in enumerate(X):
            X[i].fitness = func_vals[i]
            X[i].n_eval += 1
            if not warm_start:
                self.eval_count += 1

            self.logger.info(
                f"#{i + 1} - fitness: {func_vals[i]}, solution: {self._to_pheno(X[i])}"
            )

        X = self.post_eval_check(X)
        self.data = self.data + X if hasattr(self, "data") else X
        self.update_model()

        if self.data_file is not None:
            X.to_csv(self.data_file, header=False, append=True)

        xopt = self.xopt
        self.logger.info(f"fopt: {xopt.fitness}")
        # show the current penalty value if cheap constraints are present
        if self.h is not None or self.g is not None:
            _penalty = dynamic_penalty(xopt.tolist(), 1, self._h, self._g, minimize=self.minimize)
            self.logger.info(f"penalty: {_penalty[0]:.4e}")
        self.logger.info(f"xopt: {self._to_pheno(xopt)}\n")

        if not self.model.is_fitted:
            self._fBest_DoE = copy(xopt.fitness)  # the best point in the DoE
            self._xBest_DoE = copy(self._to_pheno(xopt))

        if not warm_start:
            self.iter_count += 1
            self.hist_f.append(xopt.fitness)

    @timeit
    def evaluate(self, X) -> List[float]:
        """Evaluate the candidate points and update evaluation info in the dataframe"""
        # Parallelization is handled by the objective function itself
        if self.parallel_obj_fun is not None:
            func_vals = self.parallel_obj_fun(X)
        else:
            if self.n_job > 1:  # or by ourselves..
                func_vals = Parallel(n_jobs=self.n_job)(delayed(self.obj_fun)(x) for x in X)
            else:  # or sequential execution
                func_vals = [self.obj_fun(x) for x in X]
        return func_vals

    def recommend(self) -> Solution:
        if self.xopt is None or len(self.xopt) == 0:
            raise RecommendationUnavailableError()
        return self.xopt

    def create_DoE(self, n_point: int, fixed: Dict = None) -> List:
        """get the initial sample points using Design of Experiemnt (DoE) methods

        Parameters
        ----------
        n_point : int
            the number of sample points to draw

        Returns
        -------
        List
            a list of sample points
        """
        fixed = {} if fixed is None else fixed
        search_space = self.search_space.filter(fixed.keys(), invert=True)

        count = 0
        DoE = []
        while n_point:
            # NOTE: random sampling could generate duplicated points again
            # keep sampling until getting enough points
            X = search_space.sample(
                n_point,
                method="LHS" if n_point > 1 else "uniform",
                h=partial_argument(self._h, self.search_space.var_name, fixed)
                if self._h
                else None,
                g=partial_argument(self._g, self.search_space.var_name, fixed)
                if self._g
                else None,
            ).tolist()
            X = fillin_fixed_value(X, fixed, self.search_space)

            if len(X) != 0:
                X = self.search_space.round(X)
                X = self.pre_eval_check(X)
                DoE += X
                n_point -= len(X)

            count += 1
            if count > 3:  # maximally 3 iterations
                break

        return DoE

    @abstractmethod
    def pre_eval_check(self, X: Solution) -> Solution:
        """This function is meant for checking validaty of the solutions prior to the evaluation

        Parameters
        ----------
        X : Solution
            the candidate solutions

        """
        raise NotImplementedError

    def post_eval_check(self, X: Solution) -> Solution:
        _ = np.isnan(X.fitness) | np.isinf(X.fitness)
        if np.any(_):
            self.logger.warning(
                f"{sum(_)} candidate solutions are removed "
                f"due to failed fitness evaluation: \n{str(X[_, :])}"
            )
            X = X[~_, :]
        return X

    def update_model(self):
        # TODO: implement a proper model selection here
        # TODO: in case r2 is really poor, re-fit the model or log-transform `fitness`?
        data = self.data
        fitness = data.fitness

        # TODO: to standardize the response values to prevent numerical overflow that might
        # appear in the MGF-based acquisition function.
        # Standardization should make it easier to specify the GP prior, compared to
        # rescaling values to the unit interval.
        _std = np.std(fitness)
        if len(fitness) > 5 and np.isclose(_std, 0):
            raise FlatFitnessError()

        fitness_ = (
            fitness if np.isclose(_std, 0) else (fitness - np.mean(fitness)) / np.std(fitness)
        )
        self.fmin, self.fmax = np.min(fitness_), np.max(fitness_)
        self.frange = self.fmax - self.fmin

        self.model.fit(data, fitness_)
        fitness_hat = self.model.predict(data)

        r2 = r2_score(fitness_, fitness_hat)
        MAPE = mean_absolute_percentage_error(fitness_, fitness_hat)
        self.logger.info(f"model r2: {r2}, MAPE: {MAPE}")

    @timeit
    def arg_max_acquisition(
        self, n_point: int = None, return_value: bool = False, fixed: Dict = None
    ) -> List[list]:
        """Global Optimization of the acquisition function / Infill criterion

        Returns
        -------
            candidates: tuple of list,
                candidate solution (in list)
            values: tuple,
                criterion value of the candidate solution
        """
        self.logger.debug("acquisition optimziation...")
        n_point = self.n_point if n_point is None else int(n_point)
        return_dx = self._optimizer == "BFGS"
        self.__set_argmax(fixed)

        if n_point > 1:  # multi-point/batch sequential strategy
            candidates, values = self._batch_arg_max_acquisition(
                n_point=n_point, return_dx=return_dx, fixed=fixed
            )
        else:  # single-point strategy
            criteria = self._create_acquisition(par={}, return_dx=return_dx, fixed=fixed)
            candidates, values = self._argmax_restart(criteria, logger=self.logger)
            candidates, values = [candidates], [values]

        candidates = [c for c in candidates if len(c) != 0]
        candidates = fillin_fixed_value(candidates, fixed, self.search_space)
        for callback in self._acquisition_callbacks:
            callback()

        return (candidates, values) if return_value else candidates

    def _create_acquisition(
        self, fun: str = None, par: dict = None, return_dx: bool = False, fixed: Dict = None
    ) -> Callable:
        fun = fun if fun is not None else self._acquisition_fun
        par = par if par is not None else copy(self._acquisition_par)
        par.update({"model": self.model, "minimize": self.minimize})
        criterion = getattr(AcquisitionFunction, fun)(**par)
        return partial_argument(
            func=functools.partial(criterion, return_dx=return_dx),
            var_name=self.search_space.var_name,
            fixed=fixed,
            reduce_output=return_dx,
        )

    def _batch_arg_max_acquisition(self, n_point: int, return_dx: int, fixed: Dict):
        raise NotImplementedError

    def check_stop(self):
        if self.eval_count >= self.max_FEs:
            self.stop_dict["max_FEs"] = self.eval_count

        if self.ftarget is not None and self.xopt is not None:
            if self._compare(self.xopt.fitness[0], self.ftarget):
                self.stop_dict["ftarget"] = self.xopt.fitness[0]

        return bool(self.stop_dict)

    def save(self, filename: str):
        # creat the folder if not exist
        dirname = os.path.dirname(filename)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)

        with open(filename, "wb") as f:
            # NOTE: we need to dump `self.data` first. Otherwise, some
            # attributes of it will be lost
            if hasattr(self, "data"):
                self.data = dill.dumps(self.data)

            FHs = list(filter(lambda h: isinstance(h, logging.FileHandler), self.logger.handlers))
            if len(FHs) == 0:
                _logger = None
            else:
                _logger = FHs[0].baseFilename  # Only take the first log file

            logger = self.logger
            self.logger = _logger

            dill.dump(self, f)

            self.logger = logger
            if hasattr(self, "data"):
                self.data = dill.loads(self.data)
            self.logger.info(f"save to file {filename}...")

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            obj = dill.load(f)
            if hasattr(obj, "data"):
                obj.data = dill.loads(obj.data)
            obj.logger = getattr(obj, "_logger")
        return obj
