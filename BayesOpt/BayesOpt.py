# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:05:01 2017

@author: Hao Wang
@email: wangronin@gmail.com

"""
from pdb import set_trace
import os, sys, dill, functools, logging, time

import pandas as pd
import numpy as np
import json, copy, re 

from joblib import Parallel, delayed
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

from .base import Solution
from .optimizer import mies, cma_es
from .InfillCriteria import EI, PI, MGFI
from .Surrogate import SurrogateAggregation
from .misc import proportional_selection, non_dominated_set_2d, bcolors, MyFormatter

# TODO: this goes to utils.py
verbose = {
    False : logging.NOTSET,
    'DEBUG' : logging.DEBUG,
    'INFO' : logging.INFO
}

# TODO: implement the automatic surrogate model selection
# TODO: improve the efficiency; profiling
class BO(object):
    """Bayesian Optimization base class"""
    def __init__(self, search_space, obj_func, surrogate, parallel_obj_func=None, ftarget=None,
                 eq_func=None, ineq_func=None, minimize=True, max_eval=None, 
                 max_iter=None, init_points=None, warm_data=None,
                 infill='EI', t0=2, tf=1e-1, schedule='exp', eval_type='list',
                 n_init_sample=None, n_point=1, n_job=1,
                 n_restart=None, max_infill_eval=None, wait_iter=3, optimizer='MIES', 
                 data_file=None, verbose=False, random_seed=None, log_file=None):
        """

        Parameters
        ----------
            search_space : instance of SearchSpace type
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
        # TODO: clean up and split this function into sub-procedures.
        self.verbose = verbose
        self.data_file = data_file
        self.log_file = log_file
        self._space = search_space
        self.var_names = self._space.var_name
        self.obj_func = obj_func
        self.parallel_obj_func = parallel_obj_func
        self.eq_func = eq_func
        self.ineq_func = ineq_func
        self.surrogate = surrogate
        self.n_point = int(n_point)
        self.n_job = int(n_job)
        self.ftarget = ftarget 
        self.infill = infill
        self.minimize = minimize
        self.dim = len(self._space)
        self._best = min if self.minimize else max
        self._eval_type = eval_type         # TODO: find a better name for this
        self.n_obj = 1
        self.init_points = init_points
        
        self.r_index = self._space.id_C       # index of continuous variable
        self.i_index = self._space.id_O       # index of integer variable
        self.d_index = self._space.id_N       # index of categorical variable

        self.param_type = self._space.var_type
        self.N_r = len(self.r_index)
        self.N_i = len(self.i_index)
        self.N_d = len(self.d_index)

        self._init_flatfitness_trial = 2

        # parameter: objective evaluation
        # TODO: for noisy objective function, maybe increase the initial evaluations
        self.init_n_eval = 1      
        self.max_eval = int(max_eval) if max_eval else np.inf
        self.max_iter = int(max_iter) if max_iter else np.inf
        self.n_init_sample = self.dim * 20 if n_init_sample is None else int(n_init_sample)
        self.eval_hist = []
        self.eval_hist_id = []
        self.iter_count = 0
        self.eval_count = 0

        # TODO: this is just an ad-hoc fix 
        if self.n_point > 1:
            self.infill = 'MGFI'
        
        # setting up cooling schedule
        # subclassing this part
        if self.infill == 'MGFI':
            self.t0 = t0
            self.tf = tf
            self.t = t0
            self.schedule = schedule
            
            # TODO: find a nicer way to integrate this part
            # cooling down to 1e-1
            max_iter = self.max_eval - self.n_init_sample
            if self.schedule == 'exp':                         # exponential
                self.alpha = (self.tf / t0) ** (1. / max_iter) 
            elif self.schedule == 'linear':
                self.eta = (t0 - self.tf) / max_iter           # linear
            elif self.schedule == 'log':
                self.c = self.tf * np.log(max_iter + 1)        # logarithmic 
            elif self.schedule == 'self-adaptive':
                raise NotImplementedError

        # paramter: acquisition function optimziation
        mask = np.nonzero(self._space.C_mask | self._space.O_mask)[0]

        # bounds for continuous and integer variable
        self._bounds = np.array([self._space.bounds[i] for i in mask])  # continous and integer
        self._levels = np.array([self._space.bounds[i] for i in self._space.id_N]) # discrete
        self._optimizer = optimizer
        # TODO: set this _max_eval smaller when using L-BFGS and larger for MIES
        self._max_eval = int(5e2 * self.dim) if max_infill_eval is None else max_infill_eval
        self._random_start = int(5 * self.dim) if n_restart is None else n_restart
        self._wait_iter = int(wait_iter)    # maximal restarts when optimal value does not change

        # stop criteria
        self.stop_dict = {}
        self.hist_f = []
        self._check_params()

        # set the random seed
        self.random_seed = random_seed
        if self.random_seed:
            np.random.seed(self.random_seed)
        
        # setup the logger
        self.set_logger(self.log_file)

        # load initial data 
        if (warm_data is not None 
                and isinstance(warm_data, Solution)):
            self._check_var_name_consistency(warm_data.var_name)
            self.warm_data = warm_data
        elif (warm_data is not None 
                and isinstance(warm_data, str)):
            self._load_initial_data(warm_data)

    def __del__(self):
        self.logger.handlers = []   # completely de-register the logger

    def set_logger(self, logger):
        """Create the logging object
        Params:
            logger : str, None or logging.Logger,
                either a logger file name, None (no logging) or a logger object
        """
        if isinstance(logger, logging.Logger):
            self.logger = logger

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        fmt = MyFormatter()

        if self.verbose != 0:
            # create console handler and set level to warning
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

        # create file handler and set level to debug
        if logger is not None:
            fh = logging.FileHandler(logger)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

        if hasattr(self, 'logger'):
            self.logger.propagate = False
    
    def step(self):
        X = self.ask()    

        t0 = time.time()
        func_vals = self.evaluate(X)
        self.logger.info('evaluation takes {:.4f}s'.format(time.time() - t0))

        self.tell(X, func_vals)

    def run(self):
        while not self.check_stop():
            self.step()

        return self.xopt.tolist(), self.xopt.fitness, self.stop_dict

    def ask(self, n_point=None):
        if hasattr(self, 'data'):  
            n_point = self.n_point if n_point is None else min(self.n_point, n_point)

            X = self.arg_max_acquisition(n_point=n_point)[0]
            X = Solution(X, index=len(self.data) + np.arange(len(X)), var_name=self.var_names)
            
            X = self.pre_eval_check(X)
            # draw the remaining ones randomly
            if len(X) < n_point:
                self.logger.warn("iteration {}: duplicated solution found " 
                                 "by optimization! New points is taken from random "
                                 "design".format(self.iter_count))
                N = n_point - len(X)
                method = 'LHS' if N > 1 else 'uniform'
                s = self._space.sampling(N=N, method=method) 
                X = Solution(X.tolist() + s, 
                             index=len(self.data) + np.arange(len(X)), 
                             var_name=self.var_names)

        else: # initial DoE
            X = self.create_DoE(self.n_init_sample)
        
        return X
    
    def tell(self, X, func_vals):
        if not isinstance(X, Solution):
            X = Solution(X, index=len(self.data) + np.arange(len(X)), var_name=self.var_names)

        if self.iter_count == 0:
            self.logger.info(bcolors.WARNING + \
                'initial DoE of size {}:'.format(len(X)) + bcolors.ENDC)
        else:
            self.logger.info(bcolors.WARNING + \
                'iteration {}, {} infill points:'.format(self.iter_count, len(X)) + \
                    bcolors.ENDC)

        for i in range(len(X)):
            X[i].fitness = func_vals[i]
            X[i].n_eval += 1
            self.logger.info('#{} - fitness: {},  x: {}'.format(i + 1, func_vals[i], 
                             self._space.to_dict(X[i])))

        X = self.post_eval_check(X)
        self.data = self.data + X if hasattr(self, 'data') else X

        # re-train the surrogate model 
        self.update_surrogate()   

        if self.data_file is not None:
            X.to_csv(self.data_file, header=False, append=True)

        self.fopt = self._best(self.data.fitness)
        _ = np.nonzero(self.data.fitness == self.fopt)[0][0]
        self.xopt = self.data[_]   

        self.logger.info(bcolors.WARNING + \
            'fopt: {}'.format(self.fopt) + bcolors.ENDC)   
        self.logger.info(bcolors.WARNING + \
            'xopt: {}\n'.format(self._space.to_dict(self.xopt)) + bcolors.ENDC)     

        self.iter_count += 1
        self.hist_f.append(self.xopt.fitness)
        self.stop_dict['n_eval'] = self.eval_count
        self.stop_dict['n_iter'] = self.iter_count

    def create_DoE(self, n_point=None):
        DoE = [] if self.init_points is None else self.init_points

        while len(DoE) < n_point:
            DoE += self._space.sampling(n_point - len(DoE), method='LHS')
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
            self.logger.warn('{} candidate solutions are removed '
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
            X = [self._space.to_dict(x) for x in data]

        # Parallelization is handled by the objective function itself
        if self.parallel_obj_func is not None:  
            func_vals = self.parallel_obj_func(X)
        else:
            if self.n_job > 1:
                func_vals = Parallel(n_jobs=self.n_job)(delayed(self.obj_func)(x) for x in X)
            else:
                func_vals = [self.obj_func(x) for x in X]
                
        self.eval_count += N
        return func_vals

    def update_surrogate(self):
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
        self.surrogate.fit(data, fitness_scaled)
        
        fitness_hat = self.surrogate.predict(data)
        r2 = r2_score(fitness_scaled, fitness_hat)

        # TODO: adding cross validation for the model? 
        # TODO: how to prevent overfitting in this case
        # TODO: in case r2 is really poor, re-fit the model or transform the input? 
        # TODO: perform diagnostic/validation on the surrogate model
        # consider the performance metric transformation in SMAC
        self.logger.info('Surrogate model r2: {}'.format(r2))
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
        self.logger.debug('infill criteria optimziation...')
        t0 = time.time()

        n_point = self.n_point if n_point is None else min(self.n_point, n_point)
        dx = True if self._optimizer == 'BFGS' else False
        criteria = [self._acquisition(plugin, dx=dx) for i in range(n_point)]
        
        if self.n_job > 1:
            __ = Parallel(n_jobs=self.n_job)(delayed(self._argmax_multistart)(c) for c in criteria)
        else:
            __ = [list(self._argmax_multistart(_)) for _ in criteria]

        candidates, values = tuple(zip(*__))
        self.logger.debug('infill criteria optimziation takes {:.4f}s'.format(time.time() - t0))

        return candidates, values
    
    # TODO: this function should be removed
    def initialize(self):
        """Generate the initial data set (DOE) and construct the surrogate model"""
        if hasattr(self, 'data'):
            self.logger.warn('initialization is already performed!') 
            return

        if hasattr(self, 'warm_data'):
            self.logger.info('adding warm data to the initial model')
            self.data = copy.deepcopy(self.warm_data)

        self.logger.info('selected surrogate model: {}'.format(self.surrogate.__class__)) 
        self.logger.info('building {:d} initial design points...'.format(self.n_init_sample))
        
        sampling_trial = self._init_flatfitness_trial
        while True:
            DOE = []
            while len(DOE) < self.n_init_sample:
                if len(DOE) > 0:
                    self.logger.info('adding %d new points due to duplications...'
                        %(self.n_init_sample - len(DOE)))
                    DOE += Solution(self._space.sampling(self.n_init_sample - len(DOE)), 
                                    var_name=self.var_names, n_obj=self.n_obj)
                elif self.init_points is not None:
                    n = len(self.init_points)
                    DOE = self.init_points + self._space.sampling(self.n_init_sample - n)
                else:
                    DOE = self._space.sampling(self.n_init_sample)
                if not isinstance(DOE, Solution):
                    DOE = Solution(DOE, var_name=self.var_names, n_obj=self.n_obj)
                DOE = self._remove_duplicate(DOE)

            self.evaluate(DOE, runs=self.init_n_eval)
            DOE = self.after_eval_check(DOE)
            
            if hasattr(self, 'data') and len(self.data) != 0:
                self.data += DOE
            else:
                self.data = DOE

            fmin, fmax = np.min(self.data.fitness), np.max(self.data.fitness)
            if np.isclose(fmin, fmax): 
                if sampling_trial > 0:
                    self.logger.warning('flat objective value in the initialization!')
                    self.logger.warning('resampling the initial points...')
                    sampling_trial -= 1
                else:
                    self.logger.warning('flat objective value after taking {} '
                    'samples (each has {} sample points)...'.format(self._init_flatfitness_trial, 
                             self.n_init_sample))
                    self.logger.warning('optimization terminates...')

                    self.stop_dict['flatfitness'] = True
                    self.fopt = self._best(self.data.fitness)
                    _ = np.nonzero(self.data.fitness == self.fopt)[0][0]
                    self.xopt = self.data[_]  
                    return
            else:
                break

        for i, x in enumerate(DOE):
            self.logger.info('DOE {}, fitness: {} -- {}'.format(i + 1, x.fitness, 
                             self._space.to_dict(x)))

        self.fit_and_assess()
        if self.data_file is not None: # save the initial design to csv
            DOE.to_csv(self.data_file)
            #self.data.to_csv(self.data_file)

    def check_stop(self):
        # TODO: add more stop criteria
        if self.iter_count >= self.max_iter:
            self.stop_dict['max_iter'] = True

        if self.eval_count >= self.max_eval:
            self.stop_dict['max_eval'] = True
        
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
        plugin = 0 if plugin is None else (plugin - self.fmin) / self.frange
            
        if self.n_point > 1:  # multi-point method
            # create a portofolio of n infill-criteria by 
            # instantiating n 't' values from the log-normal distribution
            # exploration and exploitation
            # TODO: perhaps also introduce cooling schedule for MGF
            # TODO: other method: niching, UCB, q-EI
            tt = np.exp(self.t * np.random.randn())
            acquisition_func = MGFI(self.surrogate, plugin, minimize=self.minimize, t=tt)
            self._annealling()
            
        elif self.n_point == 1:   # sequential excution
            if self.infill == 'EI':
                acquisition_func = EI(self.surrogate, plugin, minimize=self.minimize)
            elif self.infill == 'PI':
                acquisition_func = PI(self.surrogate, plugin, minimize=self.minimize)
            elif self.infill == 'MGFI':
                # TODO: move this part to adaptive BayesOpt
                acquisition_func = MGFI(self.surrogate, plugin, minimize=self.minimize, t=self.t)
                self._annealling()
            elif self.infill == 'UCB':
                raise NotImplementedError
                
        return functools.partial(acquisition_func, dx=dx) 
    
    # TODO: this function goes to the child class 
    def _annealling(self):
        if self.schedule == 'exp':  
             self.t *= self.alpha
        elif self.schedule == 'linear':
            self.t -= self.eta
        elif self.schedule == 'log':
            # TODO: verify this
            self.t = self.c / np.log(self.iter_count + 1 + 1)
   
    def _argmax_multistart(self, obj_func):
        # keep the list of optima in each restart for future usage
        xopt, fopt = [], []  
        eval_budget = self._max_eval
        best = -np.inf
        wait_count = 0

        for iteration in range(self._random_start):
            x0 = self._space.sampling(N=1, method='uniform')[0]
            
            # TODO: add IPOP-CMA-ES here for testing
            # TODO: when the surrogate is GP, implement a GA-BFGS hybrid algorithm
            # TODO: BFGS only works with continuous parameters
            # TODO: add constraint handling for BFGS
            if self._optimizer == 'BFGS':
                if self.N_d + self.N_i != 0:
                    raise ValueError('BFGS is not supported with mixed variable types.')
                # TODO: find out why: somehow this local lambda function can be pickled...
                # for minimization
                func = lambda x: tuple(map(lambda x: -1. * x, obj_func(x)))
                xopt_, fopt_, stop_dict = fmin_l_bfgs_b(func, x0, pgtol=1e-8,
                                                        factr=1e6, bounds=self._bounds,
                                                        maxfun=eval_budget)
                xopt_ = xopt_.flatten().tolist()
                fopt_ = -np.asscalar(fopt_)
                
                if stop_dict["warnflag"] != 0:
                    self.logger.debug("L-BFGS-B terminated abnormally with the "
                                      " state: %s" % stop_dict)
                                
            elif self._optimizer == 'MIES':
                opt = mies(self._space, obj_func, eq_func=self.eq_func, ineq_func=self.ineq_func,
                           max_eval=eval_budget, minimize=False, verbose=False, 
                           eval_type=self._eval_type)                           
                xopt_, fopt_, stop_dict = opt.optimize()

            if fopt_ > best:
                best = fopt_
                wait_count = 0
                self.logger.debug('restart : {} - funcalls : {} - Fopt : {}'.format(iteration + 1, 
                                   stop_dict['funcalls'], fopt_))
            else:
                wait_count += 1

            eval_budget -= stop_dict['funcalls']
            xopt.append(xopt_)
            fopt.append(fopt_)
            
            if eval_budget <= 0 or wait_count >= self._wait_iter:
                break

        # maximization: sort the optima in descending order
        idx = np.argsort(fopt)[::-1]
        return xopt[idx[0]], fopt[idx[0]]

    def _check_params(self):
        if np.isinf(self.max_eval) and np.isinf(self.max_iter):
            raise ValueError('max_eval and max_iter cannot be both infinite')

    def _check_var_name_consistency(self, var_name):
        if len(self._space.var_name) == len(var_name):
            for a,b in zip(self._space.var_name, var_name):
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
            self.logger.info(str(len(self.warm_data)) + " points loaded from " + filename)
        except IOError:
            raise Exception("the " + filename + " does not contain a valid set of solutions")


# TODO: remove this part, which should be covered by the examples
if __name__ == '__main__':
    from .SearchSpace import ContinuousSpace, OrdinalSpace, NominalSpace
    from .Surrogate import RandomForest

    np.random.seed(666)

    if 11 < 2: # test for flat fitness
        def fitness(x):
            return 1

        space = ContinuousSpace([-5, 5]) * 2
        levels = space.levels if hasattr(space, 'levels') else None
        model = RandomForest(levels=levels)

        opt = BO(space, fitness, model, max_eval=300, verbose=True, n_job=1, n_point=1)
        print(opt.run())

    if 1 < 2:
        def fitness1(x):
            x_r, x_i, x_d = np.array(x[:2]), x[2], x[3]
            if x_d == 'OK':
                tmp = 0
            else:
                tmp = 1
            return np.sum(x_r ** 2) + abs(x_i - 10) / 123. + tmp * 2

        space = (ContinuousSpace([-5, 5]) * 2) + OrdinalSpace([5, 15]) + \
            NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])

        levels = space.levels if hasattr(space, 'levels') else None
        model = RandomForest(levels=levels)

        opt = BO(space, fitness1, model, max_eval=300, verbose=True, n_job=1, n_point=3,
                 n_init_sample=3,
                 init_points=[[0, 0, 10, 'OK']])
        xopt, fopt, stop_dict = opt.run()