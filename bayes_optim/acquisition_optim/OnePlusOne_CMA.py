from pdb import set_trace
import logging, sys
import numpy as np

from scipy.linalg import solve_triangular
from typing import Callable, Any, Tuple, List, Union
from ..misc import handle_box_constraint, LoggerFormatter

Vector = List[float]
Matrix = List[Vector]

def set_bounds(bound, dim):
    if isinstance(bound, str):
        bound = eval(bound)
    elif isinstance(bound, (float, int)):
        bound = [bound] * dim
    elif hasattr(bound, '__iter__'):
        bound = list(bound)
        if len(bound) == 1:
            bound *= dim
    assert len(bound) == dim
    return np.asarray(bound)

class OnePlusOne_CMA(object):
    def __init__(
        self, 
        dim: int, 
        obj_fun: Callable,
        x0: Union[str, Vector, np.ndarray] = None,
        sigma0: Union[float] = None, 
        C0: Union[Matrix, np.ndarray] = None, 
        lb: Union[float, str, Vector, np.ndarray] = -np.inf,
        ub: Union[float, str, Vector, np.ndarray] = np.inf,
        ftarget: Union[int, float] = np.inf,
        max_FEs: Union[int, str] = np.inf, 
        minimize: bool = True,
        opts: dict = {},
        verbose: bool = False,
        logger = None
        ):

        self.dim = dim
        self.obj_fun = obj_fun
        self.ftarget = ftarget
        self.minimize = minimize
        self.lb = set_bounds(lb, self.dim)
        self.ub = set_bounds(ub, self.dim)
        self.sigma0 = self.sigma = sigma0

        self.x = x0
        self.max_FEs = int(eval(max_FEs)) if isinstance(max_FEs, str) else max_FEs
        self._init_aux_var(opts)
        self._init_covariance(C0)

        self.eval_count = 0
        self.iter_count = 0
        self.xopt = self._x
        self.fopt = np.inf
        self.stop_dict = {}
        self._exception = False
        self.verbose = verbose
        self.logger = logger
        
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
        
    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, sigma):
        if not sigma:
            assert all(~np.isinf(self.lb)) & all(~np.isinf(self.ub))
            sigma = np.max(self.ub - self.lb) / 5
        assert sigma > 0
        self._sigma = sigma

    def run(self):
        while not self.check_stop():
            self.step()
            
        return self.xopt, self.fopt, self.stop_dict
            
    def step(self):
        x = self.ask()
        y = self.evaluate(x)
        self.tell(x, y)
    
    def evaluate(self, x):
        return self.obj_fun(x)

    def ask(self, n_point=1) -> np.ndarray:
        """The mutation function

        Parameters
        ----------
        n_point : int, optional
            the number of mutants, by default 1

        Returns
        -------
        np.ndarray
            [description]
        """
        z = np.random.randn(self.dim)
        x = self._x + self.sigma * z.dot(self._A.T)
        x = handle_box_constraint(x, self.lb, self.ub) 
        return x

    def tell(self, x, y):
        if hasattr(y, '__iter__'):
            y = y[0]
            
        success = y < self.fopt
        z = (x - self._x) / self._sigma
        self._update_step_size(success)

        if success:
            self.fopt = y
            self._x = self.xopt = x
            self._update_covariance(z)

        self._handle_exception()
        self.eval_count += 1
        self.iter_count += 1

        if self.verbose:
            self._logger.info('iteration {},'.format(self.eval_count))
            self._logger.info('fopt: {}'.format(self.fopt)) 
            self._logger.info('sigma: {}'.format(self._sigma)) 
            self._logger.info('xopt: {}\n'.format(self.xopt.tolist()))

    def check_stop(self):
        if self.fopt <= self.ftarget:
            self.stop_dict['ftarget'] = self.fopt
            
        if self.eval_count >= self.max_FEs:
            self.stop_dict['FEs'] = self.eval_count

        return bool(self.stop_dict)

    def _update_covariance(self, z):
        if self.success_rate < self.threshold:
            self.pc = (1 - self.cc) * self.pc + np.sqrt(self._coeff) * z
            self._C = (1 - self.ccov) * self._C + self.ccov * np.outer(self.pc, self.pc)
        else:
            self.pc = (1 - self.cc) * self.pc
            self._C = (1 - self.ccov * (1 - self._coeff)) * self._C + \
                self.ccov * np.outer(self.pc, self.pc)

        self._C = np.triu(self._C) + np.triu(self._C, 1).T 
        self._update_A(self._C)

    def _update_step_size(self, success):
        self.success_rate = (1 - self.cp) * self.success_rate + self.cp * success
        self.sigma *= np.exp(
            (self.success_rate - self.prob_target) / (1 - self.prob_target) / self.d
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
        if self._sigma < 1e-16 or self._sigma > 1e16:
            self._exception = 1
        
        if self._exception:
            self._C = np.eye(self.dim)
            self.pc = np.zeros((self.dim, 1))
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

    def _handle_exception(self):
        pass