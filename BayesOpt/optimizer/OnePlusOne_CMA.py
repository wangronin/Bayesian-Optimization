from pdb import set_trace
import numpy as np

from scipy.linalg import solve_triangular
from typing import Callable, Any, Tuple, List, Union
from ..misc import handle_box_constraint

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
        ftarget: Union[int, float] = -np.inf,
        max_FEs: Union[int, str] = np.inf, 
        opts: dict = {},
        verbose: bool = False
        ):

        self.dim = dim
        self.sigma0 = self.sigma = sigma0
        self.obj_fun = obj_fun
        self.ftarget = ftarget
        self.lb = set_bounds(lb, self.dim)
        self.ub = set_bounds(ub, self.dim)

        self.x = x0
        self.max_FEs = int(eval(max_FEs)) if isinstance(max_FEs, str) else max_FEs
        self._init_covariance(C0)
        self._init_aux_var(opts)

        self.eval_count = 0
        self.iter_count = 0
        self.xopt = self._x
        self.fopt, self._y = np.inf, np.inf
        self.stop_dict = {}
        self._exception = False
        self.verbose = verbose
        
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

    def _init_covariance(self, C=None):
        if C:
            try:
                A = np.linalg.cholesky(C)
                if np.any(~np.isreal(A)):
                    reset = True
                else:
                    self.A = A
                    self.C = C
            except np.linalg.LinAlgError:
                reset = True
                
        if C is None or reset:
            self.C = np.eye(self.dim)
            self.A = np.eye(self.dim)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        if x:
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
        assert sigma > 0
        self._sigma = sigma

    def run(self):
        while self.check_stop():
            self.step()
            
        return self.xopt, self.fopt, self.stop_dict
            
    def step(self):
        x = self.ask()
        y = self.obj_fun(x)
        self.tell(x, y)
        
    def ask(self) -> np.ndarray:
        z = np.random.randn(self.dim)
        x = self._x + self.sigma * z.dot(self.A.T)
        x = handle_box_constraint(x, self.lb, self.ub) 
        return x

    def tell(self, x, y):
        success = y < self._y
        z = (x - self._x) / self._sigma
        self._update_step_size(success)

        if success:
            self._y = self.fopt = y
            self._x = self.xopt = x
            self._update_covariance(z)

        self._handle_exception()

        self.eval_count += 1
        self.iter_count += 1

        if self.verbose:
            print('FEs {}: fopt -- {}, sigma -- {}'.format(
                self.eval_count, self.fopt, self._sigma)
            )

    def check_stop(self):
        if self.fopt <= self.ftarget:
            self.stop_dict['ftarget'] = self.fopt
            
        if self.eval_count >= self.max_FEs:
            self.stop_dict['FEs'] = self.eval_count

        return not bool(self.stop_dict)

    def _update_covariance(self, z):
        if self.success_rate < self.threshold:
            self.pc = (1 - self.cc) * self.pc + np.sqrt(self._coeff) * z
            self.C = (1 - self.ccov) * self.C + self.ccov * np.outer(self.pc, self.pc)
        else:
            self.pc = (1 - self.cc) * self.pc
            self.C = (1 - self.ccov * (1 - self._coeff)) * self.C + \
                self.ccov * np.outer(self.pc, self.pc)

        self.C = np.triu(self.C) + np.triu(self.C, 1).T 
        self._update_A(self.C)

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
                    self.A = A
            except np.linalg.LinAlgError:
                self._exception = True

    def _handle_exception(self):
        if self._sigma < 1e-16 or self._sigma > 1e16:
            self._exception = 1
        
        if self._exception:
            self.C = np.eye(self.dim)
            self.pc = np.zeros((self.dim, 1))
            self.A = np.eye(self.dim)
            self._sigma = self.sigma0
            self._exception = False

class OnePlusOne_Cholesky_CMA(OnePlusOne_CMA):
    def _init_covariance(self, C=None):
        if C:
            try:
                A = np.linalg.cholesky(C)
                if np.any(~np.isreal(A)):
                    reset = True
                else:
                    self.A_inv = solve_triangular(A, np.eye(self.dim), lower=True)
                    self.A = A
            except np.linalg.LinAlgError:
                reset = True

        if C is None or reset:
            self.A = np.eye(self.dim)
            self.A_inv = np.eye(self.dim)

    def _update_covariance(self, z):
        cb = self.ccov
        if self.success_rate < self.threshold:
            self.pc = (1 - self.cc) * self.pc + np.sqrt(self._coeff) * z
            ca = (1 - self.ccov)
        else:
            self.pc = (1 - self.cc) * self.pc
            ca = (1 - self.ccov) + self.ccov * self.cc * (2 - self.cc)

        w = self.pc.dot(self.A_inv.T)
        w_ = w.dot(self.A_inv)
        L = np.sum(w ** 2)
        
        self.A += (np.sqrt(1 + L * cb / ca) - 1) / L * np.outer(self.pc, w)
        self.A *= np.sqrt(ca)

        self.A_inv -= (1 - 1 / np.sqrt(1 + L * cb / ca)) / L * np.outer(w, w_)
        self.A_inv *= 1 / np.sqrt(ca)

    def _handle_exception(self):
        pass