import logging, sys
from typing import Callable, Any, Tuple, List, Union, Dict

import numpy as np
from scipy.linalg import solve_triangular

from ..utils import dynamic_penalty, set_bounds
from ..misc import handle_box_constraint, LoggerFormatter

Vector = List[float]
Matrix = List[Vector]

class OnePlusOne_CMA(object):
    def __init__(
        self,
        dim: int,
        obj_fun: Callable,
        args: Tuple = (),
        h: Callable = None,
        g: Callable = None,
        x0: Union[str, Vector, np.ndarray] = None,
        sigma0: Union[float] = None,
        C0: Union[Matrix, np.ndarray] = None,
        lb: Union[float, str, Vector, np.ndarray] = -np.inf,
        ub: Union[float, str, Vector, np.ndarray] = np.inf,
        ftarget: Union[int, float] = None,
        max_FEs: Union[int, str] = np.inf,
        minimize: bool = True,
        n_restart: int = 0,
        xtol: float = 1e-4,
        ftol: float = 1e-4,
        verbose: bool = False,
        logger: str = None,
        **kwargs
        ):
        """ (1+1)-CMA-ES and (1+1)-Cholesky-CMA-ES
        Hereafter, we use the following customized types to describe the usage:

        - Vector = List[float]
        - Matrix = List[Vector]

        Parameters
        ----------
        dim : int
            Dimensionality of the search space.
        obj_fun : Callable
            The objective function to be minimized.
        args: Tuple
            The extra parameters passed to function `obj_fun.`
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
        """
        self.dim = dim
        self.obj_fun = obj_fun
        self.h = h
        self.g = g
        self.minimize = minimize
        self.ftarget = ftarget
        self.lb = set_bounds(lb, self.dim)
        self.ub = set_bounds(ub, self.dim)
        self.sigma0 = self.sigma = sigma0
        self.args = args
        self.n_restart = max(0, int(n_restart))

        self.eval_count = 0
        self.iter_count = 0
        self.max_FEs = int(eval(max_FEs)) if isinstance(max_FEs, str) else max_FEs
        self._better = (lambda a, b: a <= b) if self.minimize else (lambda a, b: a >= b)
        self._init_aux_var(kwargs)
        self._init_covariance(C0)
        self._init_logging_var()

        self.x = x0
        self.stop_dict: Dict = {}
        self._exception = False
        self.verbose = verbose
        self.logger = logger

        # parameters for stopping criteria
        # NOTE: `self.xtol * 200` and `self._w = 0.9` lead to a tolerance of
        # ~50 iterations of stagnation.
        self.xtol = xtol
        self.ftol = ftol
        self._delta_x = self.xtol * 200
        self._delta_f = self.ftol * 200
        self._w = 0.9
        self._stop = False

    def _init_aux_var(self, opts):
        self.prob_target = opts['p_succ_target'] if 'p_succ_target' in opts else 2 / 11
        self.threshold = opts['p_threshold'] if 'p_threshold' in opts else 0.44
        self.d = opts['d'] if 'd' in opts else 1 + self.dim / 2
        self.ccov = opts['ccov'] if 'ccov' in opts else 2 / (self.dim ** 2 + 6)
        self.cp = opts['cp'] if 'cp' in opts else 1 / 12
        self.cc = opts['cc'] if 'cc' in opts else 2 / (self.dim + 2)

        self.success_rate = self.prob_target
        self.pc = np.zeros(self.dim)
        self._coeff = self.cc * (2 - self.cc)

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

        if self.verbose:
            # create console handler and set level to warning
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(fmt)
            self._logger.addHandler(ch)

        # create file handler and set level to debug
        if logger is not None:
            fh = logging.FileHandler(logger)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

        if hasattr(self, 'logger'):
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
            assert all(~np.isinf(self.lb)) & all(~np.isinf(self.ub))
            x = (self.ub - self.lb) * np.random.rand(self.dim) + self.lb
        self._x = x

        y = self.evaluate(x)
        _y = y + dynamic_penalty(
            x, self.iter_count + 1,
            self.h, self.g,
            minimize=self.minimize
        )

        if not hasattr(self, 'fopt_penalized') \
            or self._better(_y, self.fopt_penalized):
            self.fopt = y
            self.fopt_penalized = _y
            self.xopt = self._x

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
        self.tell(x, y)

        self.logging()
        self.check_stop()
        self.restart()

    def evaluate(self, x):
        return self.obj_fun(x, *self.args)

    def restart(self):
        if self._restart:
            self._logger.info('restarting... ')
            self.x = None
            self.sigma = self.sigma0
            self.pc = np.zeros(self.dim)
            self._C = np.eye(self.dim)
            self._A = np.eye(self.dim)
            self._delta_x = np.ones(10) * self.xtol * 10
            self._delta_f = np.ones(10) * self.ftol * 10
            self.stop_dict = {}
            self.n_restart -= 1

    def ask(self, n_point=1) -> np.ndarray:
        """The mutation function

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
        z = np.random.randn(self.dim)
        x = self._x + self.sigma * z.dot(self._A.T)
        x = handle_box_constraint(x, self.lb, self.ub)
        return x

    def tell(self, x, y):
        # NOTE: should the `tell` function also respect the stopping criteria?
        if self._stop:
            self._logger(
                'The optimizer is stopped and `tell` should not be called.'
            )

        if hasattr(y, '__iter__'):
            y = y[0]

        # TODO: decide whether the constraint handling should be part of
        # the tell function
        _y = y + dynamic_penalty(
            x, self.iter_count + 1,
            self.h, self.g,
            minimize=self.minimize
        )

        success = self._better(_y, self.fopt_penalized)
        z = (x - self._x) / self._sigma
        self._update_step_size(success)

        self._delta_f *= self._w
        self._delta_x *= self._w
        if success:
            self._delta_f += (1 - self._w) * (self.fopt_penalized - _y)
            self._delta_x += (1 - self._w) * np.sqrt(np.sum((self.xopt - x) ** 2))
            self.fopt = y
            self.fopt_penalized = _y
            self._x = self.xopt = x
            self._update_covariance(z)

        self._handle_exception()
        self.eval_count += 1
        self.iter_count += 1

        if self.verbose:
            self._logger.info('iteration {}'.format(self.eval_count))
            self._logger.info('fopt: {}'.format(self.fopt))
            self._logger.info('sigma: {}'.format(self._sigma))
            self._logger.info('xopt: {}\n'.format(self.xopt.tolist()))

    def logging(self):
        # TODO: the function name is a bit off.. since we have self._logger
        self.hist_fopt += [self.fopt]
        self.hist_xopt += [self.xopt.tolist()]
        self.hist_fopt_penalized += [self.fopt_penalized]
        self._hist_delta_x += [self._delta_x]
        self._hist_delta_f += [self._delta_f]

    def check_stop(self):
        if self.ftarget is not None and self._better(self.fopt, self.ftarget):
            self.stop_dict['ftarget'] = self.fopt

        if self.eval_count >= self.max_FEs:
            self.stop_dict['FEs'] = self.eval_count

        # TODO: add this as an option: lower and upper bounds for regular sigmas
        if self.sigma < 1e-8 or self.sigma > 1e8:
            self.stop_dict['sigma'] = self.sigma

        if self._delta_f < self.ftol:
            self.stop_dict['ftol'] = self._delta_f

        if self._delta_x < self.xtol:
            self.stop_dict['xtol'] = self._delta_x

        if 'ftarget' in self.stop_dict or 'FEs' in self.stop_dict:
            self._stop = True
        else:
            self._stop = bool(self.stop_dict) and self.n_restart == 0
            self._restart = False
            # TODO: fix the restaring criteria and turn this on
            # self._restart = bool(self.stop_dict) and self.n_restart > 0

    def _update_covariance(self, z):
        if self.success_rate < self.threshold:
            self.pc = (1 - self.cc) * self.pc + np.sqrt(self._coeff) * z
            self._C = (1 - self.ccov) * self._C + \
                self.ccov * np.outer(self.pc, self.pc)
        else:
            self.pc = (1 - self.cc) * self.pc
            self._C = (1 - self.ccov * (1 - self._coeff)) * self._C + \
                self.ccov * np.outer(self.pc, self.pc)

        self._C = np.triu(self._C) + np.triu(self._C, 1).T
        self._update_A(self._C)

    def _update_step_size(self, success):
        prob_target = self.prob_target
        self.success_rate = (1 - self.cp) * self.success_rate + \
            self.cp * success
        self._sigma *= np.exp(
            (self.success_rate - prob_target) / (1 - prob_target) / self.d
        )

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

        # TODO: maybe ad-hoc fix here can be improved?
        if self._exception:
            self._C = np.eye(self.dim)
            self.pc = np.zeros(self.dim)
            self._A = np.eye(self.dim)
            self._sigma = self.sigma0
            self._exception = False

class OnePlusOne_Cholesky_CMA(OnePlusOne_CMA):
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
            ca = (1 - self.ccov)
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