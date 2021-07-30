import logging
import sys
from copy import copy
from typing import Callable, Dict, List, Union

import numpy as np
from scipy.linalg import solve_triangular

from ..misc import LoggerFormatter, handle_box_constraint
from ..search_space import RealSpace, SearchSpace
from ..utils import dynamic_penalty, set_bounds

Vector = List[float]
Matrix = List[Vector]

__authors__ = ["Hao Wang"]


class OnePlusOne_CMA(object):
    """(1+1)-CMA-ES"""

    def __init__(
        self,
        search_space: SearchSpace,
        obj_fun: Callable,
        args: Dict = None,
        h: Callable = None,
        g: Callable = None,
        x0: Union[str, Vector, np.ndarray] = None,
        sigma0: Union[float] = None,
        C0: Union[Matrix, np.ndarray] = None,
        ftarget: Union[int, float] = None,
        max_FEs: Union[int, str] = np.inf,
        minimize: bool = True,
        n_restart: int = 0,
        xtol: float = 1e-4,
        ftol: float = 1e-4,
        verbose: bool = False,
        logger: str = None,
        random_seed: int = 42,
        **kwargs,
    ):
        """Hereafter, we use the following customized
        types to describe the usage:

        - Vector = List[float]
        - Matrix = List[Vector]

        Parameters
        ----------
        dim : int
            Dimensionality of the search space.
        obj_fun : Callable
            The objective function to be minimized.
        args: Tuple
            The extra parameters passed to function `obj_fun`.
        h : Callable, optional
            The equality constraint function, by default None.
        g : Callable, optional
            The inequality constraint function, by default None.
        x0 : Union[str, Vector, np.ndarray], optional
            The initial guess (by default None) which must fall between lower
            and upper bounds, if non-infinite values are provided for `lb` and
            `ub`. Note that, `x0` must be provided when `lb` and `ub` both
            take infinite values.
        sigma0 : Union[float], optional
            The initial step size, by default None
        C0 : Union[Matrix, np.ndarray], optional
            The initial covariance matrix which must be positive definite,
            by default None. Any non-positive definite input will be ignored.
        lb : Union[float, str, Vector, np.ndarray], optional
            The lower bound of search variables. When it is not a `float`,
            it must have the same length as `upper`, by default `-np.inf`.
        ub : Union[float, str, Vector, np.ndarray], optional
            The upper bound of search variables. When it is not a `float`,
            it must have the same length as `lower`, by default `np.inf`.
        ftarget : Union[int, float], optional
            The target value to hit, by default None.
        max_FEs : Union[int, str], optional
            Maximal number of function evaluations to make, by default `np.inf`.
        minimize : bool, optional
            To minimize or maximize, by default True.
        xtol : float, optional
            Absolute error in xopt between iterations that is acceptable for
            convergence, by default 1e-4.
        ftol : float, optional
            Absolute error in func(xopt) between iterations that is acceptable
            for convergence, by default 1e-4.
        n_restart : int, optional
            The maximal number of random restarts to perform when stagnation is
            detected during the run. The random restart can be switched off by
            setting `n_restart` to zero (the default value).
        verbose : bool, optional
            Verbosity of the output, by default False.
        logger : str, optional
            Name of the logger file, by default None, which turns off the
            logging behaviour.
        random_seed : int, optional
            The seed for pseudo-random number generators, by default None.
        """
        assert isinstance(search_space, RealSpace)
        lb, ub = list(zip(*search_space.bounds))
        self.search_space = search_space
        self.dim: int = search_space.dim
        self.obj_fun: Callable = obj_fun
        self.h: Callable = h
        self.g: Callable = g
        self.minimize: bool = minimize
        self.ftarget: float = ftarget
        self.lb: np.ndarray = set_bounds(lb, self.dim)
        self.ub: np.ndarray = set_bounds(ub, self.dim)
        self.sigma = sigma0
        self.sigma0 = self.sigma
        self.args: Dict = args if args else {}
        self.n_restart: int = max(0, int(n_restart))
        self._restart: bool = False

        self.xopt: np.ndarray = None
        self.fopt: float = None
        self.fopt_penalized: float = None
        self.eval_count: int = 0
        self.iter_count: int = 0
        self.max_FEs: int = int(eval(max_FEs)) if isinstance(max_FEs, str) else max_FEs
        self._better = (lambda a, b: a <= b) if self.minimize else (lambda a, b: a >= b)
        self._init_aux_var(kwargs)
        self._init_covariance(C0)
        self._init_logging_var()

        self.stop_dict: Dict = {}
        self._exception: bool = False
        self.verbose: bool = verbose
        self.logger = logger
        self.random_seed = random_seed

        # parameters for stopping criteria
        # NOTE: `self._delta_f = self.ftol / self._w ** (5 * self.dim)`
        # and `self._w = 0.9` lead to a tolerance of
        # ~`5 * self.dim` iterations of stagnation.
        self.xtol: float = xtol
        self.ftol: float = ftol
        self._w: float = 0.9
        self._delta_x: float = self.xtol / self._w ** (5 * self.dim)
        self._delta_f: float = self.ftol / self._w ** (5 * self.dim)
        self._stop: bool = False
        # set the initial search point
        self.x = x0

    def _init_aux_var(self, opts):
        self.prob_target = opts["p_succ_target"] if "p_succ_target" in opts else 2 / 11
        self.threshold = opts["p_threshold"] if "p_threshold" in opts else 0.44
        self.d = opts["d"] if "d" in opts else 1 + self.dim / 2
        self.ccov = opts["ccov"] if "ccov" in opts else 2 / (self.dim ** 2 + 6)
        self.cp = opts["cp"] if "cp" in opts else 1 / 12
        self.cc = opts["cc"] if "cc" in opts else 2 / (self.dim + 2)

        self.success_rate: float = self.prob_target
        self.pc: np.ndarray = np.zeros(self.dim)
        self._coeff: float = self.cc * (2 - self.cc)

    def _init_covariance(self, C):
        if C is None:
            self._C = np.eye(self.dim)
            self._A = np.eye(self.dim)
        else:
            self.C = C

    def _init_logging_var(self):
        # parameters for logging the history
        self.hist_fopt: List = []
        self.hist_fopt_penalized: List = []
        self.hist_xopt: List = []
        self._hist_delta_x: List = []
        self._hist_delta_f: List = []

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
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        if isinstance(logger, logging.Logger):
            self._logger = logger
            self._logger.propagate = False
            return

        self._logger = logging.getLogger(self.__class__.__name__)
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

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, C):
        if C is not None:
            try:
                A = np.linalg.cholesky(C)
                if np.all(np.isreal(A)):
                    # TODO: `_A` should be a private attribute
                    self._A = A
                    self._C = C
            except np.linalg.LinAlgError:
                pass

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        if x is not None:
            x = eval(x) if isinstance(x, str) else x
            x = np.asarray(x)
            assert np.all(x - self.lb >= 0)
            assert np.all(x - self.ub <= 0)
        else:
            # sample `x` u.a.r. in `[lb, ub]`
            assert all(~np.isinf(self.lb)) & all(~np.isinf(self.ub))
            x = (self.ub - self.lb) * np.random.rand(self.dim) + self.lb

        self._x = x
        y = self.evaluate(x)
        penalty = self.penalize(x)
        self.tell(x, y, penalty)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        if sigma is None:
            assert all(~np.isinf(self.lb)) & all(~np.isinf(self.ub))
            sigma = np.max(self.ub - self.lb) / 5
        assert sigma > 0
        self._sigma = sigma

    def run(self):
        while not self._stop:
            self.step()
        return self.xopt, self.fopt, self.stop_dict

    def step(self):
        x = self.ask()
        y = self.evaluate(x)
        self.tell(x, y, self.penalize(x))
        self.logging()
        self.check_stop()
        self.restart()

    def penalize(self, x: np.ndarray):
        """Calculate the dynamic penalty once the constraint functions are provided

        Parameters
        ----------
        x : np.ndarray
            the trial point to check against the constraints
        """
        return dynamic_penalty(x, self.iter_count + 1, self.h, self.g, minimize=self.minimize)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        self.eval_count += 1
        if isinstance(self.args, (list, tuple)):
            fval = self.obj_fun(x, *self.args)
        elif isinstance(self.args, dict):
            fval = self.obj_fun(x, **self.args)
        return fval

    def restart(self):
        if self._restart:
            self.logger.info("restarting... ")
            self.x = None
            self.sigma = self.sigma0
            self.pc = np.zeros(self.dim)
            self._C = np.eye(self.dim)
            self._A = np.eye(self.dim)
            self._delta_x = self.xtol * 200
            self._delta_f = self.ftol * 200
            self.stop_dict = {}
            self.n_restart -= 1

    def ask(self) -> np.ndarray:
        """The mutation operator

        Parameters
        ----------
        n_point : int, optional
            The number of mutants, which is always 1. This argument is only
            meant to keep the function interface consistant.

        Returns
        -------
        np.ndarray
            The mutation vector
        """
        z = np.random.randn(self.dim).dot(self._A.T)
        x = self._x + self.sigma * z
        x = handle_box_constraint(x, self.lb, self.ub)
        # rounding if a coarser numerical precision is provided
        x = self.search_space.round(x).ravel()
        # NOTE: experimental correction to the step-size when the box constraints are violated
        # self.sigma = np.min(np.abs((x - self._x) / z))
        return x

    def tell(self, x: np.ndarray, y: np.ndarray, penalty: float = 0):
        if self._stop:
            self.logger.info("The optimizer is stopped and `tell` should not be called.")
            return

        # TODO: this might not be uncessary
        if hasattr(y, "__iter__"):
            y = y[0]
        if hasattr(penalty, "__iter__"):
            penalty = penalty[0]
        y_penalized = y + penalty

        if self.xopt is None:
            self.fopt = y
            self.fopt_penalized = y_penalized
            self.xopt = x
            return

        success = self._better(y_penalized, self.fopt_penalized)
        z = (x - self._x) / self._sigma
        self._update_step_size(success)
        self._delta_f *= self._w
        self._delta_x *= self._w

        if success:
            self._delta_f += (1 - self._w) * abs(self.fopt_penalized - y_penalized)
            self._delta_x += (1 - self._w) * np.sqrt(sum((self._x - x) ** 2))
            self.fopt_penalized = y_penalized
            self._x = copy(x)
            self._update_covariance(z)

        if success and penalty == 0:
            self.xopt = copy(self._x)
            self.fopt = y

        self._handle_exception()
        self.iter_count += 1

        if self.verbose:
            self.logger.info(f"iteration {self.iter_count}")
            self.logger.info(f"fopt: {self.fopt}")
            if self.h is not None or self.g is not None:
                _penalty = (self.fopt - self.fopt_penalized) * (-1) ** self.minimize
                self.logger.info(f"penalty: {_penalty[0]:.4e}")
            self.logger.info(f"xopt: {self.xopt.tolist()}")
            self.logger.info(f"sigma: {self._sigma}\n")

    def logging(self):
        self.hist_fopt += [self.fopt]
        self.hist_xopt += [self.xopt.tolist()]
        self.hist_fopt_penalized += [self.fopt_penalized]

    def check_stop(self):
        if self.ftarget is not None and self._better(self.fopt, self.ftarget):
            self.stop_dict["ftarget"] = self.fopt

        if self.eval_count >= self.max_FEs:
            self.stop_dict["FEs"] = self.eval_count

        # TODO: add this as an option: lower and upper bounds for regular sigmas
        if self.sigma < 1e-8 or self.sigma > 1e8:
            self.stop_dict["sigma"] = self.sigma

        if self._delta_f < self.ftol:
            self.stop_dict["ftol"] = self._delta_f

        if self._delta_x < self.xtol:
            self.stop_dict["xtol"] = self._delta_x

        if "ftarget" in self.stop_dict or "FEs" in self.stop_dict:
            self._stop = True
        else:
            if self.n_restart > 0:
                self._restart = bool(self.stop_dict)
            else:
                self._stop = bool(self.stop_dict)

    def _update_covariance(self, z):
        if self.success_rate < self.threshold:
            self.pc = (1 - self.cc) * self.pc + np.sqrt(self._coeff) * z
            self._C = (1 - self.ccov) * self._C + self.ccov * np.outer(self.pc, self.pc)
        else:
            self.pc = (1 - self.cc) * self.pc
            self._C = (1 - self.ccov * (1 - self._coeff)) * self._C + self.ccov * np.outer(
                self.pc, self.pc
            )

        self._C = np.triu(self._C) + np.triu(self._C, 1).T
        self._update_A(self._C)

    def _update_step_size(self, success):
        prob_target = self.prob_target
        self.success_rate = (1 - self.cp) * self.success_rate + self.cp * success
        self._sigma *= np.exp((self.success_rate - prob_target) / (1 - prob_target) / self.d)
        if self._sigma is None:
            breakpoint()

    def _update_A(self, C):
        if np.any(np.isinf(C)):
            self._exception = True
        else:
            try:
                A = np.linalg.cholesky(C)
                if np.any(~np.isreal(A)):
                    self._exception = True
                else:
                    self._A = A
            except np.linalg.LinAlgError:
                self._exception = True

    def _handle_exception(self):
        if self._sigma < 1e-8 or self._sigma > 1e8:
            self._exception = 1

        if self._exception:
            self._C = np.eye(self.dim)
            self.pc = np.zeros(self.dim)
            self._A = np.eye(self.dim)
            self._sigma = self.sigma0
            self._exception = False


class OnePlusOne_Cholesky_CMA(OnePlusOne_CMA):
    """(1+1)-Cholesky-CMA-ES improves its base class algorithm by taking advantage of
    Cholesky's decomposition to update the covariance, which is computationally cheaper

    """

    def _init_covariance(self, C):
        reset = False
        if C is not None:
            try:
                A = np.linalg.cholesky(C)
                if np.any(~np.isreal(A)):
                    reset = True
                else:
                    self.A = A
            except np.linalg.LinAlgError:
                reset = True

        if C is None or reset:
            self.A = np.eye(self.dim)

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, A):
        assert np.all(np.triu(A, k=1).ravel() == 0)
        self._A = A
        self._A_inv = solve_triangular(A, np.eye(self.dim), lower=True)

    def _update_covariance(self, z):
        cb = self.ccov
        if self.success_rate < self.threshold:
            self.pc = (1 - self.cc) * self.pc + np.sqrt(self._coeff) * z
            ca = 1 - self.ccov
        else:
            self.pc = (1 - self.cc) * self.pc
            ca = (1 - self.ccov) + self.ccov * self.cc * (2 - self.cc)

        w = self.pc.dot(self._A_inv.T)
        w_ = w.dot(self._A_inv)
        L = np.sum(w ** 2)

        self._A += (np.sqrt(1 + L * cb / ca) - 1) / L * np.outer(self.pc, w)
        self._A *= np.sqrt(ca)

        self._A_inv -= (1 - 1 / np.sqrt(1 + L * cb / ca)) / L * np.outer(w, w_)
        self._A_inv *= 1 / np.sqrt(ca)
