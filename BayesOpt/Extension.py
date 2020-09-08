import logging, sys
import numpy as np

from typing import Callable
from .misc import LoggerFormatter

class OptimizerPipeline(object):
    def __init__(
        self,
        obj_fun: Callable,
        n_point: int = 1,
        ftarget: float = -np.inf,
        max_FEs: int = None,
        minimize: bool = True,
        verbose: bool = False,
        logger: str = None
        ):
        self.obj_fun = obj_fun
        self.max_FEs = max_FEs
        self.ftarget = ftarget
        self.n_point = n_point
        self.verbose = verbose
        self.minimize = minimize
        self.queue = []
        self.stop_dict = []
        self.N = 0
        self.logger = logger
        self._counter = 0
        self._curr_opt = None
        self._transfer = None
        self._stop = False

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
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

    def add(self, opt, transfer=None):
        opt.obj_fun = self.obj_fun
        opt.verbose = self.verbose
        opt.ftarget = self.ftarget
        opt.minimize = self.minimize
        opt.max_FEs = self.max_FEs
        opt.logger = self._logger

        # add pairs of (optimizer, transfer function)
        self.queue.append((opt, transfer)) 
        self.N += 1

    def ask(self, n_point=None):
        """Get suggestions from the optimizer.

        Parameters
        ----------
        n_point : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        if not self._curr_opt:
            self._curr_opt, self._transfer = self.queue[self._counter]
            self._logger.name = self._curr_opt.__class__.__name__
        
        n_point = n_point if n_point else self.n_point
        return self._curr_opt.ask(n_point=n_point)

    def tell(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        # Update the model with new objective function observations
        # ...
        # No return statement needed
        self._curr_opt.tell(X, y)
        self.switch()
        self.xopt = self._curr_opt.xopt
        self.fopt = self._curr_opt.fopt

    def switch(self):
        """To check if the current optimizer is stopped
        and switch to the next optimizer
        """
        if self._curr_opt.check_stop():
            _max_FEs = self.max_FEs - self._curr_opt.eval_count
            _counter = self._counter + 1

            if _counter < self.N and _max_FEs > 0:
                _curr_opt, _transfer = self.queue[_counter]
                _curr_opt.max_FEs = _max_FEs
                _curr_opt.xopt = np.array(self._curr_opt.xopt)
                _curr_opt.fopt = self._curr_opt.fopt

                if self._transfer:
                    kwargs = self._transfer(self._curr_opt)
                    for k, v in kwargs.items():
                        setattr(_curr_opt, k, v)

                self._logger.name = _curr_opt.__class__.__name__
                self._counter = _counter
                self._curr_opt = _curr_opt
                self._transfer = _transfer
            else:
                self._stop = True
            
    def evaluate(self, X):
        if not hasattr(X[0], '__iter__'):
            X = [X]
        return [self.obj_fun(x) for x in X]

    def step(self):
        X = self.ask()    
        func_vals = self.evaluate(X)
        self.tell(X, func_vals)

    def run(self):
        while not self._stop:
            self.step()
        return self.xopt, self.fopt, self.stop_dict