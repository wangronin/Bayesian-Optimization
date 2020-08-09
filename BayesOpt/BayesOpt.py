# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:16:39 2018

@author: Hao Wang
@email: wangronin@gmail.com

"""
from pdb import set_trace

from typing import Callable, Any, Tuple
import os, sys, dill, functools, logging, time

import pandas as pd
import numpy as np
import json, copy, re 

from joblib import Parallel, delayed
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

from .base import Solution
from .SearchSpace import SearchSpace
from .optimizer import mies, cma_es
from .InfillCriteria import EI, PI, MGFI
from .Surrogate import SurrogateAggregation
from .misc import proportional_selection, non_dominated_set_2d, bcolors, MyFormatter

# To use `dill` for the pickling, which works for
# much more python objects
os.environ['LOKY_PICKLER'] = 'dill' 

# TODO: this goes to utils.py
verbose = {
    False : logging.NOTSET,
    'DEBUG' : logging.DEBUG,
    'INFO' : logging.INFO
}


# TODO: move this to utility
def arg_to_int(arg):
    if isinstance(arg, str):
        x = int(eval(arg))
    elif isinstance(arg, (int, float)):
        x = int(arg)
    else: 
        raise ValueError
    return x

# def get_default_param():
#     info =  {
#             'optimizer' : None,
#             'n_restart': None, 
#             'max_FEs': None, 
#             'wait_iter': 3 
#         }

class baseBO(object):
    def __init__(
        self, 
        search_space: SearchSpace, 
        obj_fun: Callable,
        parallel_obj_fun: Callable = None,
        eq_fun: Callable = None, 
        ineq_fun: Callable = None, 
        model = None,
        eval_type: str = 'list',
        DoE_size: int = None, 
        n_point: int = 1,
        acquisition_fun: str = 'EI',
        acquisition_optimization: dict = {},
        ftarget: float = None,
        max_FEs: int = None, 
        minimize: bool = True, 
        n_job: int = 1,
        data_file: str = None, 
        verbose: bool = False, 
        random_seed: int = None, 
        logger: str = None,
        ):
        """[summary]

        Parameters
        ----------
        search_space : SearchSpace
            The search space, an instance of `SearchSpace` class
        obj_fun : Callable
            [description]
        parallel_obj_fun : Callable, optional
            [description], by default None
        eq_fun : Callable, optional
            [description], by default None
        ineq_fun : Callable, optional
            [description], by default None
        model : [type], optional
            [description], by default None
        eval_type : str, optional
            [description], by default 'list'
        DoE_size : int, optional
            [description], by default None
        n_point : int, optional
            [description], by default 1
        acquisition_fun : str, optional
            [description], by default 'EI'
        acquisition_optimization : dict, optional
            [description], by default {}
        ftarget : float, optional
            [description], by default None
        max_FEs : int, optional
            [description], by default None
        minimize : bool, optional
            [description], by default True
        n_job : int, optional
            [description], by default 1
        data_file : str, optional
            [description], by default None
        verbose : bool, optional
            [description], by default False
        random_seed : int, optional
            [description], by default None
        logger : str, optional
            [description], by default None
        """        

        """

        Parameters
        ----------
            search_space : 
            obj_func : callable,
                the objective function to optimize
            surrogate: surrogate model, currently support either GPR or random forest
            minimize : bool,
                minimize or maximize
            max_eval : int,
                maximal number of evaluations on the objective function
            max_iter : int,
                maximal iteration
            eval_type : str,
                type of arguments to be evaluated: list | dict  
            n_init_sample : int,
                the size of inital Design of Experiment (DoE),
                default: 20 * dim
            n_point : int,
                the number of candidate solutions proposed using infill-criteria,
                default : 1
            n_job : int,
                the number of jobs scheduled for parallelizing the evaluation. 
                Only Effective when n_point > 1 
            optimizer: str,
                the optimization algorithm for infill-criteria,
                supported options are:
                    'MIES' (Mixed-Integer Evolution Strategy), 
                    'BFGS' (quasi-Newtion for GPR)
        """
        self.obj_fun = obj_fun
        self.parallel_obj_fun = parallel_obj_fun
        self.h = eq_fun
        self.g = ineq_fun
        self.n_job = max(1, int(n_job))
        self.n_point = max(1, int(n_point))
        self.ftarget = ftarget 
        self.minimize = minimize
        self.verbose = verbose
        self.data_file = data_file

        self.search_space = search_space
        self.DoE_size = DoE_size
        self.acquisition_fun = acquisition_fun
        self.model = model
        self.logger = logger
        self.random_seed = random_seed
        self._set_AQ_optimization(**acquisition_optimization)
        self._set_aux_vars()
        
        self.max_FEs = int(max_FEs) if max_FEs else np.inf      
        self.n_obj = 1
        self._get_best = min if self.minimize else max
        self._eval_type = eval_type   
        self._init_flatfitness_trial = 2
        self._check_params()
    
    @property
    def acquisition_fun(self):
        return self.__AQ_fun

    @acquisition_fun.setter
    def acquisition_fun(self, fun):
        if isinstance(fun, str):
            self.__AQ_fun = fun
        else:
            assert hasattr(fun, '__call__')
        self.__AQ_fun = fun

    @property
    def DoE_size(self):
        return self.__DoE_size

    @DoE_size.setter
    def DoE_size(self, DoE_size):
        if DoE_size:
            if isinstance(DoE_size, str):
                self.__DoE_size = int(eval(DoE_size))
            elif isinstance(DoE_size, (int, float)):
                self.__DoE_size = int(DoE_size)
            else: 
                raise ValueError
        else:
            self.__DoE_size = int(self.dim * 20)

    @property
    def random_seed(self):
        return self.__random_seed
    
    @random_seed.setter
    def random_seed(self, seed):
        if seed:
            self.__random_seed = int(seed)
            if self.__random_seed:
                np.random.seed(self.__random_seed)

    @property
    def search_space(self):
        return self.__search_space

    @search_space.setter
    def search_space(self, search_space):
        self.__search_space = search_space
        self.dim = len(self.__search_space)
        self.var_names = self.__search_space.var_name
        self.r_index = self.__search_space.id_C       # indices of continuous variable
        self.i_index = self.__search_space.id_O       # indices of integer variable
        self.d_index = self.__search_space.id_N       # indices of categorical variable

        self.param_type = self.__search_space.var_type
        self.N_r = len(self.r_index)
        self.N_i = len(self.i_index)
        self.N_d = len(self.d_index)

        mask = np.nonzero(self.__search_space.C_mask | self.__search_space.O_mask)[0]
        self._bounds = np.array([self.__search_space.bounds[i] for i in mask]) 
        self._levels = np.array(
            [self.__search_space.bounds[i] for i in self.__search_space.id_N]
        ) 

    @property
    def logger(self):
        return self.__logger

    @logger.setter
    def logger(self, logger):
        """Create the logging object
        Params:
            logger : str, None or logging.Logger,
                either a logger file name, None (no logging) or a logger object
        """
        if isinstance(logger, logging.Logger):
            self.__logger = logger
            self.__logger.propagate = False
            return

        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.setLevel(logging.DEBUG)
        fmt = MyFormatter()

        if self.verbose != 0:
            # create console handler and set level to warning
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(fmt)
            self.__logger.addHandler(ch)

        # create file handler and set level to debug
        if logger is not None:
            fh = logging.FileHandler(logger)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            self.__logger.addHandler(fh)

        if hasattr(self, 'logger'):
            self.__logger.propagate = False
    
    def _set_aux_vars(self):
        self.iter_count = 0
        self.eval_count = 0
        self.eval_hist = []
        self.eval_hist_id = []
        self.stop_dict = {}
        self.hist_f = []

    def _set_AQ_optimization(self, optimizer, max_FEs, n_restart, wait_iter):
        self.__optimizer = optimizer

        if max_FEs is None:
            self.AQ_max_FEs = int(5e2 * self.dim) if self.__optimizer == 'MIES' else \
                int(1e2 * self.dim)
        else:
            self.AQ_max_FEs = arg_to_int(max_FEs)
        
        self.AQ_n_restart = int(5 * self.dim) if n_restart is None else arg_to_int(n_restart)
        self.AQ_wait_iter = 3 if wait_iter is None else arg_to_int(wait_iter)

    def run(self):
        while not self.check_stop():
            self.step()

        return self.xopt.tolist(), self.xopt.fitness, self.stop_dict

    def step(self):
        X = self.ask()    

        t0 = time.time()
        func_vals = self.evaluate(X)
        self.__logger.info('evaluation takes {:.4f}s'.format(time.time() - t0))

        self.tell(X, func_vals)

    def ask(self, n_point=None):
        if hasattr(self, 'data'):  
            n_point = self.n_point if n_point is None else min(self.n_point, n_point)

            X = self.arg_max_acquisition(n_point=n_point)[0]
            X = self.__search_space.round(X)
            X = Solution(
                X, index=len(self.data) + np.arange(len(X)), 
                var_name=self.var_names
            )
            
            X = self.pre_eval_check(X)

            # TODO: handle the constrains when performing random sampling
            # draw the remaining ones randomly
            if len(X) < n_point:
                self.__logger.warn(
                    "iteration {}: duplicated solution found " 
                    "by optimization! New points is taken from random "
                    "design".format(self.iter_count)
                )
                N = n_point - len(X)
                method = 'LHS' if N > 1 else 'uniform'
                s = self.__search_space.sampling(N=N, method=method) 
                X = X.tolist() + s
                X = Solution(
                    X, index=len(self.data) + np.arange(len(X)), 
                    var_name=self.var_names
                )

        else: # initial DoE
            X = self.create_DoE(self.__DoE_size)
        
        return X
    
    def tell(self, X, func_vals):
        if not isinstance(X, Solution):
            X = Solution(X, var_name=self.var_names)

        msg = 'initial DoE of size {}:'.format(len(X)) if self.iter_count == 0 else \
            'iteration {}, {} infill points:'.format(self.iter_count, len(X))
        self.__logger.info(msg)

        for i in range(len(X)):
            X[i].fitness = func_vals[i]
            X[i].n_eval += 1
            self.__logger.info(
                '#{} - fitness: {},  solution: {}'.format(i + 1, func_vals[i], 
                self.__search_space.to_dict(X[i]))
            )

        X = self.post_eval_check(X)
        self.data = self.data + X if hasattr(self, 'data') else X

        # re-train the surrogate model 
        self.update_surrogate()   

        if self.data_file is not None:
            X.to_csv(self.data_file, header=False, append=True)

        self.fopt = self._get_best(self.data.fitness)
        _ = np.nonzero(self.data.fitness == self.fopt)[0][0]
        self.xopt = self.data[_]   

        self.__logger.info('fopt: {}'.format(self.fopt))   
        self.__logger.info('xopt: {}\n'.format(self.__search_space.to_dict(self.xopt)))     

        self.iter_count += 1
        self.hist_f.append(self.xopt.fitness)
        self.stop_dict['n_eval'] = self.eval_count
        self.stop_dict['n_iter'] = self.iter_count

    def create_DoE(self, n_point=None):
        DoE = []
        # DoE = [] if self.init_points is None else self.init_points

        while len(DoE) < n_point:
            DoE += self.__search_space.sampling(n_point - len(DoE), method='LHS')
            DoE = self.pre_eval_check(DoE).tolist()
        
        return Solution(DoE, var_name=self.var_names)

    def pre_eval_check(self, X):
        """
        check for the duplicated solutions, as it is not allowed
        for noiseless objective functions
        """
        if not isinstance(X, Solution):
            X = Solution(X, var_name=self.var_names)
        
        N = X.N
        if hasattr(self, 'data'):
            X = X + self.data

        _ = []
        for i in range(N):
            x = X[i]
            idx = np.arange(len(X)) != i
            CON = np.all(np.isclose(np.asarray(X[idx][:, self.r_index], dtype='float'),
                                    np.asarray(x[self.r_index], dtype='float')), axis=1)
            INT = np.all(X[idx][:, self.i_index] == x[self.i_index], axis=1)
            CAT = np.all(X[idx][:, self.d_index] == x[self.d_index], axis=1)
            if not any(CON & INT & CAT):
                _ += [i]

        return X[_]
    
    def post_eval_check(self, X):
        _ = np.isnan(X.fitness) | np.isinf(X.fitness)
        if np.any(_):
            if len(_.shape) == 2:  # for multi-objective cases
                _ = np.any(_, axis=1).ravel()
            self.__logger.warn('{} candidate solutions are removed '
                             'due to falied fitness evaluation: \n{}'.format(sum(_), str(X[_, :])))
            X = X[~_, :] 

        return X
        
    def evaluate(self, data):
        """Evaluate the candidate points and update evaluation info in the dataframe
        """
        N = len(data)
        if self._eval_type == 'list':
            X = [x.tolist() for x in data]
        elif self._eval_type == 'dict':
            X = [self.__search_space.to_dict(x) for x in data]

        # Parallelization is handled by the objective function itself
        if self.parallel_obj_fun is not None:  
            func_vals = self.parallel_obj_fun(X)
        else:
            if self.n_job > 1:
                func_vals = Parallel(n_jobs=self.n_job)(delayed(self.obj_fun)(x) for x in X)
            else:
                func_vals = [self.obj_fun(x) for x in X]
                
        self.eval_count += N
        return func_vals

    def update_surrogate(self):
        # TODO: implement the automatic surrogate model selection
        # adding the warm-start data when fitting the surrogate model
        data = self.data + self.warm_data if hasattr(self, 'warm_data') else self.data
        fitness = data.fitness

        # normalization the response for the numerical stability
        # e.g., for MGF-based acquisition function
        self.fmin, self.fmax = np.min(fitness), np.max(fitness)

        # flat_fitness = np.isclose(self.fmin, self.fmax)
        fitness_scaled = (fitness - self.fmin) / (self.fmax - self.fmin)
        self.frange = self.fmax - self.fmin

        # fit the surrogate model
        self.model.fit(data, fitness_scaled)
        
        fitness_hat = self.model.predict(data)
        r2 = r2_score(fitness_scaled, fitness_hat)

        # TODO: adding cross validation for the model? 
        # TODO: how to prevent overfitting in this case
        # TODO: in case r2 is really poor, re-fit the model or transform the input? 
        # TODO: perform diagnostic/validation on the surrogate model
        # consider the performance metric transformation in SMAC
        self.__logger.info('Surrogate model r2: {}'.format(r2))
        return r2

    def arg_max_acquisition(self, plugin=None, n_point=None):
        """
        Global Optimization of the acqusition function / Infill criterion
        Returns
        -------
            candidates: tuple of list,
                candidate solution (in list)
            values: tuple,
                criterion value of the candidate solution
        """
        self.__logger.debug('infill criteria optimziation...')
        t0 = time.time()

        n_point = self.n_point if n_point is None else min(self.n_point, n_point)
        dx = True if self.__optimizer == 'BFGS' else False
        criteria = [self._acquisition(plugin, dx=dx) for i in range(n_point)]
        
        if self.n_job > 1:
            __ = Parallel(n_jobs=self.n_job)(
                delayed(self._argmax_multistart)(c) for c in criteria
            )
        else:
            __ = [list(self._argmax_multistart(_)) for _ in criteria]

        candidates, values = tuple(zip(*__))
        self.__logger.debug(
            'infill criteria optimziation takes {:.4f}s'.format(time.time() - t0)
        )

        return candidates, values

    def check_stop(self):
        if self.eval_count >= self.max_FEs:
            self.stop_dict['max_FEs'] = True
        
        if self.ftarget is not None and hasattr(self, 'xopt'):
            if self._compare(self.xopt.fitness, self.ftarget):
                self.stop_dict['ftarget'] = True

        return any([v for v in self.stop_dict.values() if isinstance(v, bool)])

    def _compare(self, f1, f2):
        """Test if objecctive value f1 is better than f2
        """
        return f1 < f2 if self.minimize else f2 > f1

    def _acquisition(self, plugin=None, dx=False):
        """
        plugin : float,
            the minimal objective value used in improvement-based infill criteria
            Note that it should be given in the original scale
        """
        # objective values are normalized
        if self.noisy:
            # use the model prediction to determine the plugin under noisy scenarios
            if plugin is None:
                plugin = min(self.model.predict(self.data)) \
                    if self.minimize else max(self.model.predict(self.data))
        else:
            plugin = 0 if plugin is None else (plugin - self.fmin) / self.frange
            
        if self.n_point > 1:  # multi-point method
            # create a portofolio of n infill-criteria by 
            # instantiating n 't' values from the log-normal distribution
            # exploration and exploitation
            # TODO: perhaps also introduce cooling schedule for MGF
            # TODO: other method: niching, UCB, q-EI
            tt = np.exp(self.t * np.random.randn())
            acquisition_func = MGFI(self.model, plugin, minimize=self.minimize, t=tt)
            
        elif self.n_point == 1:   # sequential excution
            if self.acquisition_fun == 'EI':
                acquisition_func = EI(self.model, plugin, minimize=self.minimize)
            elif self.acquisition_fun == 'PI':
                acquisition_func = PI(self.model, plugin, minimize=self.minimize)
            elif self.acquisition_fun == 'MGFI':
                # TODO: move this part to adaptive BayesOpt
                acquisition_func = MGFI(self.model, plugin, 
                                        minimize=self.minimize, t=self.t)
            elif self.acquisition_fun == 'UCB':
                raise NotImplementedError
                
        return functools.partial(acquisition_func, dx=dx) 
    
    def _argmax_multistart(self, obj_func):
        # lists of the best solutions and acquisition values
        # from each restart
        xopt, fopt = [], []  
        eval_budget = self.AQ_max_FEs
        best = -np.inf
        wait_count = 0

        for iteration in range(self.AQ_n_restart):
            x0 = self.__search_space.sampling(N=1, method='uniform')[0]
            
            # TODO: when the surrogate is GP, implement a GA-BFGS hybrid algorithm
            # TODO: BFGS only works with continuous parameters
            # TODO: add constraint handling for BFGS
            if self.__optimizer == 'BFGS':
                if self.N_d + self.N_i != 0:
                    raise ValueError('BFGS is not supported with mixed variable types.')

                func = lambda x: tuple(map(lambda x: -1. * x, obj_func(x)))
                xopt_, fopt_, stop_dict = fmin_l_bfgs_b(
                    func, x0, pgtol=1e-8, factr=1e6, 
                    bounds=self._bounds, maxfun=eval_budget
                )

                xopt_ = xopt_.flatten().tolist()
                fopt_ = -np.asscalar(fopt_)
                
                if stop_dict["warnflag"] != 0:
                    self.__logger.debug(
                        "L-BFGS-B terminated abnormally with the state: %s"%stop_dict
                    )
                                
            elif self.__optimizer == 'MIES':
                opt = mies(
                    self.__search_space, obj_func, 
                    eq_func=self.h, 
                    ineq_func=self.g,
                    max_eval=eval_budget, 
                    minimize=False, 
                    verbose=False, 
                    eval_type=self._eval_type
                )                           
                xopt_, fopt_, stop_dict = opt.optimize()

            if fopt_ > best:
                best = fopt_
                wait_count = 0
                self.__logger.debug(
                    'restart : %d - funcalls : %d - Fopt : %f'%(iteration + 1, 
                    stop_dict['funcalls'], fopt_)
                )
            else:
                wait_count += 1

            eval_budget -= stop_dict['funcalls']
            xopt.append(xopt_)
            fopt.append(fopt_)
            
            if eval_budget <= 0 or wait_count >= self.AQ_wait_iter:
                break

        # maximization: sort the optima in descending order
        idx = np.argsort(fopt)[::-1]
        return xopt[idx[0]], fopt[idx[0]]

    def _check_params(self):
        if np.isinf(self.max_FEs):
            raise ValueError('max_FEs cannot be infinite')

    def _check_var_name_consistency(self, var_name):
        if len(self.__search_space.var_name) == len(var_name):
            for a,b in zip(self.__search_space.var_name, var_name):
                if a != b:
                    raise Exception("Var name inconsistency (" + str(a) + ", " + str(b) +")")
        else:
            raise Exception("Search space dim does not mathc with the warm data file")

    def _load_initial_data(self, filename, sep=","):
        try:
            var_name = None
            index_pos = 0
            with open(filename, "r") as f:
                line = f.readline()
                while line:
                    line = line.replace("\n", "").split(sep)
                    if var_name is None:
                        var_name = line
                        fitness_pos = line.index("fitness")
                        n_eval_pos = line.index("n_eval")
                        var_pos = min(fitness_pos, n_eval_pos)
                        var_name.remove("fitness")
                        var_name.remove("n_eval")
                        if line[0] == "":
                           index_pos = 1
                           var_name.remove("")
                        # Check the consistency of the csv file and the search space
                        self._check_var_name_consistency(var_name)
                    elif len(var_name) > 0:
                        var = [int(p) if p.isnumeric() else float(p) if re.match(r"^\d+?\.\d+?$", p) else p for p in line[index_pos:var_pos]]
                        sol = Solution(var, var_name=var_name, fitness=float(line[fitness_pos]), n_eval=int(line[n_eval_pos]))
                        if hasattr(self, 'warm_data') and len(self.warm_data) > 0:
                            self.warm_data += sol
                        else:
                            self.warm_data = sol
                    line = f.readline()
            f.close()
            self.__logger.info(str(len(self.warm_data)) + " points loaded from " + filename)
        except IOError:
            raise Exception("the " + filename + " does not contain a valid set of solutions")
    
    def save(self, filename):
        if hasattr(self, 'data'):
            self.data = dill.dumps(self.data)

        with open(filename, 'wb') as f:
            dill.dump(self, f)
        
        if hasattr(self, 'data'):
            self.data = dill.loads(self.data)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            obj = dill.load(f)
            if hasattr(obj, 'data'):
                obj.data = dill.loads(obj.data)
                
        return obj

class NoisyBO(BO):
    def __init__(self):
        pass