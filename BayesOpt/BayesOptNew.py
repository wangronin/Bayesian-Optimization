# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:05:01 2017

@author: wangronin
@email: wangronin@gmail.com

"""
from __future__ import division
from __future__ import print_function

import pdb
import dill, functools, itertools, copyreg, logging

import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import r2_score

from .InfillCriteria import EI, PI, MGFI
from .optimizer import mies
from .utils import proportional_selection

# TODO: remove the usage of pandas here change it to customized np.ndarray
# TODO: finalize the logging system
class Solution(np.ndarray):
    def __new__(cls, x, fitness=None, n_eval=0, index=None, var_name=None):
        obj = np.asarray(x, dtype='object').view(cls)
        obj.fitness = fitness
        obj.n_eval = n_eval
        obj.index = index
        obj.var_name = var_name
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        # Needed for array slicing
        self.fitness = getattr(obj, 'fitness', None)
        self.n_eval = getattr(obj, 'n_eval', None)
        self.index = getattr(obj, 'index', None)
        self.var_name = getattr(obj, 'var_name', None)
    
    def to_dict(self):
        if self.var_name is None: return
        return {k : self[i] for i, k in enumerate(self.var_name)}     
    
    def __str__(self):
        return self.to_dict()
    
    
class BayesOpt(object):
    """
    Generic Bayesian optimization algorithm
    """
    def __init__(self, search_space, obj_func, surrogate, ftarget=None,
                 minimize=True, noisy=False, max_eval=None, max_iter=None, 
                 infill='EI', t0=2, tf=1e-1, schedule=None,
                 n_init_sample=None, n_point=1, n_job=1, backend='multiprocessing',
                 n_restart=None, max_infill_eval=None, wait_iter=3, optimizer='MIES', 
                 log_file=None, data_file=None, verbose=False, random_seed=None):
        """
        parameter
        ---------
            search_space : instance of SearchSpace type
            obj_func : callable,
                the objective function to optimize
            surrogate: surrogate model, currently support either GPR or random forest
            minimize : bool,
                minimize or maximize
            noisy : bool,
                is the objective stochastic or not?
            max_eval : int,
                maximal number of evaluations on the objective function
            max_iter : int,
                maximal iteration
            n_init_sample : int,
                the size of inital Design of Experiment (DoE),
                default: 20 * dim
            n_point : int,
                the number of candidate solutions proposed using infill-criteria,
                default : 1
            n_job : int,
                the number of jobs scheduled for parallelizing the evaluation. 
                Only Effective when n_point > 1 
            backend : str, 
                the parallelization backend, supporting: 'multiprocessing', 'MPI', 'SPARC'
            optimizer: str,
                the optimization algorithm for infill-criteria,
                supported options: 'MIES' (Mixed-Integer Evolution Strategy), 
                                   'BFGS' (quasi-Newtion for GPR)
        """
        self.verbose = verbose
        self.log_file = log_file
        self.data_file = data_file
        self._space = search_space
        self.var_names = self._space.var_name.tolist()
        self.obj_func = obj_func
        self.noisy = noisy
        self.surrogate = surrogate
        self.n_point = n_point
        self.n_jobs = min(self.n_point, n_job)
        self._parallel_backend = backend
        self.ftarget = ftarget 
        self.infill = infill
        self.minimize = minimize
        self.dim = len(self._space)
        self._best = min if self.minimize else max
        
        self.r_index = self._space.id_C       # index of continuous variable
        self.i_index = self._space.id_O       # index of integer variable
        self.d_index = self._space.id_N       # index of categorical variable

        self.param_type = self._space.var_type
        self.N_r = len(self.r_index)
        self.N_i = len(self.i_index)
        self.N_d = len(self.d_index)
       
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
        
        # setting up cooling schedule
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
        self._bounds = np.array([self._space.bounds[i] for i in mask])             # bounds for continuous and integer variable
        # self._levels = list(self._space.levels.values())
        self._levels = np.array([self._space.bounds[i] for i in self._space.id_N]) # levels for discrete variable
        self._optimizer = optimizer
        # TODO: set this number smaller when using L-BFGS and larger for MIES
        self._max_eval = int(5e2 * self.dim) if max_infill_eval is None else max_infill_eval
        self._random_start = int(5 * self.dim) if n_restart is None else n_restart
        self._wait_iter = int(wait_iter)    # maximal restarts when optimal value does not change

        # Intensify: the number of potential configuations compared against the current best
        # self.mu = int(np.ceil(self.n_init_sample / 3))
        self.mu = 3
        
        # stop criteria
        self.stop_dict = {}
        self.hist_f = []
        self._check_params()

        # set the random seed
        self.random_seed = random_seed
        if self.random_seed:
            np.random.seed(self.random_seed)
        
        self._get_logger(self.log_file)
        
        # allows for pickling the objective function 
        copyreg.pickle(self._eval_one, dill.pickles) 
    
    def _get_logger(self, logfile):
        """
        When logfile is None, no records are written
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('- %(asctime)s [%(levelname)s] -- '
                                      '[- %(process)d - %(name)s] %(message)s')

        # create console handler and set level to warning
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # create file handler and set level to debug
        if logfile is not None:
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def _compare(self, f1, f2):
        """
        Test if perf1 is better than perf2
        """
        if self.minimize:
            return f1 < f2
        else:
            return f2 > f2
    
    def _remove_duplicate(self, data):
        """
        check for the duplicated solutions, as it is not allowed
        for noiseless objective functions
        """
        ans = []
        X = np.array([s.tolist() for s in self.data], dtype='object')
        for i, x in enumerate(data):
            CON = np.all(np.isclose(np.asarray(X[:, self.r_index], dtype='float'),
                                    np.asarray(x[self.r_index], dtype='float')), axis=1)
            INT = np.all(X[:, self.i_index] == x[self.i_index], axis=1)
            CAT = np.all(X[:, self.d_index] == x[self.d_index], axis=1)
            if not any(CON & INT & CAT):
                ans.append(x)
        return ans

    def _eval_one(self, x, runs=1):
        """
        evaluate one solution
        """
        # TODO: sometimes the obj_func take a dictionary as input...
        fitness_, n_eval = x.fitness, x.n_eval
        # try:
            # ans = [self.obj_func(x.tolist()) for i in range(runs)]
        # except:
        ans = [self.obj_func(x.to_dict()) for i in range(runs)]
        fitness = np.sum(ans)

        x.n_eval += runs
        x.fitness = fitness / runs if fitness_ is None else (fitness_ * n_eval + fitness) / x.n_eval

        self.eval_count += runs
        self.eval_hist += ans
        self.eval_hist_id += [x.index] * runs
        
        return x, runs, ans, [x.index] * runs

    def evaluate(self, data, runs=1):
        """ Evaluate the candidate points and update evaluation info in the dataframe
        """
        if isinstance(data, Solution):
            self._eval_one(data)
        
        elif isinstance(data, list): 
            if self.n_jobs > 1:
                if self._parallel_backend == 'multiprocessing': # parallel execution using joblib
                    res = Parallel(n_jobs=self.n_jobs, verbose=False)(
                        delayed(self._eval_one, check_pickle=False)(x) for x in data)
                    
                    x, runs, hist, hist_id = zip(*res)
                    self.eval_count += sum(runs)
                    self.eval_hist += list(itertools.chain(*hist))
                    self.eval_hist_id += list(itertools.chain(*hist_id))
                    for i, k in enumerate(data):
                        data[i] = x[i].copy()
                elif self._parallel_backend == 'MPI': # parallel execution using MPI
                    # TODO: to use InstanceRunner here
                    pass
                elif self._parallel_backend == 'Spark': # parallel execution using Spark
                    pass        
            else:
                for x in data:
                    self._eval_one(x)

    def fit_and_assess(self):
        X = np.atleast_2d([s.tolist() for s in self.data])
        fitness = np.array([s.fitness for s in self.data])

        # normalization the response for numerical stability
        # e.g., for MGF-based acquisition function
        _min, _max = np.min(fitness), np.max(fitness)
        fitness_scaled = (fitness - _min) / (_max - _min)

        # fit the surrogate model
        self.surrogate.fit(X, fitness_scaled)
        
        self.is_update = True
        fitness_hat = self.surrogate.predict(X)
        r2 = r2_score(fitness_scaled, fitness_hat)

        # TODO: in case r2 is really poor, re-fit the model or transform the input? 
        # consider the performance metric transformation in SMAC
        self.logger.info('Surrogate model r2: {}'.format(r2))
        return r2

    def select_candidate(self):
        # always generate mu + 1 candidate solutions
        # while True:
        #     X, infill_value = self.arg_max_acquisition()
            
        #     if self.n_point > 1:
        #         X = [Solution(x, name=len(self.data) + i) for i, x in enumerate(X)]
        #     else:
        #         X = [Solution(X, name=len(self.data))]
        #     X = self._remove_duplicate(X)

        #     # if no new design site is found, re-estimate the parameters immediately
        #     # TODO: maybe remove this rule
        #     if len(X) < self.n_point:
        #         if not self.is_update:
        #             # Duplication are commonly encountered in the 'corner'
        #             self.fit_and_assess()
        #         else:
        #             self.logger.warn("iteration {}: duplicated solution found " 
        #                              "by optimization! New points is taken from random "
        #                              "design".format(self.iter_count))
        #             N = self.n_point - len(X)
        #             if N > 1:
        #                 X = self._space.sampling(N=N, method='LHS')
        #             else:  # To generate a single sample, only uniform sampling is allowed
        #                 X = self._space.sampling(N=1, method='uniform')
        #             X = [Solution(x, name=len(self.data) + i) for i, x in enumerate(X)]
        #             break
        #     else:
        #         break
        self.is_update = False
        X, infill_value = self.arg_max_acquisition()
        
        if self.n_point > 1:
            X = [Solution(x, index=len(self.data) + i, var_name=self.var_names) for i, x in enumerate(X)]
        else:
            X = [Solution(X, index=len(self.data), var_name=self.var_names)]
            
        X = self._remove_duplicate(X)
        # if the number of new design sites obtained is less than required,
        # draw the remaining ones randomly
        if len(X) < self.n_point:
            self.logger.warn("iteration {}: duplicated solution found " 
                                "by optimization! New points is taken from random "
                                "design".format(self.iter_count))
            N = self.n_point - len(X)
            if N > 1:
                s = self._space.sampling(N=N, method='LHS')
            else:      # To generate a single sample, only uniform sampling is feasible
                s = self._space.sampling(N=1, method='uniform')
            X += [Solution(x, index=len(self.data) + i, var_name=self.var_names) for i, x in enumerate(s)]
        
        candidates_id = [x.index for x in X]
        # for noisy fitness: perform a proportional selection from the evaluated ones   
        if self.noisy:
            id_, fitness = zip([(i, d.fitness) for i, d in enumerate(self.data) if i != self.incumbent_id])
            __ = proportional_selection(fitness, self.mu, self.minimize, replacement=False)
            candidates_id.append(id_[__])
        
        # TODO: postpone the evaluate to intensify...
        self.evaluate(X, runs=self.init_n_eval)
        self.data += X
        return candidates_id

    def intensify(self, candidates_ids):
        """
        intensification procedure for noisy observations (from SMAC)
        """
        # TODO: verify the implementation here
        maxR = 20 # maximal number of the evaluations on the incumbent
        for i, ID in enumerate(candidates_ids):
            r, extra_run = 1, 1
            conf = self.data.loc[i]
            self.evaluate(conf, 1)
            print(conf.to_frame().T)

            if conf.n_eval > self.incumbent_id.n_eval:
                self.incumbent_id = self.evaluate(self.incumbent_id, 1)
                extra_run = 0

            while True:
                if self._compare(self.incumbent_id.perf, conf.perf):
                    self.incumbent_id = self.evaluate(self.incumbent_id, 
                                                   min(extra_run, maxR - self.incumbent_id.n_eval))
                    print(self.incumbent_id.to_frame().T)
                    break
                if conf.n_eval > self.incumbent_id.n_eval:
                    self.incumbent_id = conf
                    if self.verbose:
                        print('[DEBUG] iteration %d -- new incumbent selected:' % self.iter_count)
                        print('[DEBUG] {}'.format(self.incumbent_id))
                        print('[DEBUG] with performance: {}'.format(self.incumbent_id.perf))
                        print()
                    break

                r = min(2 * r, self.incumbent_id.n_eval - conf.n_eval)
                self.data.loc[i] = self.evaluate(conf, r)
                print(self.conf.to_frame().T)
                extra_run += r
    
    def _initialize(self):
        """Generate the initial data set (DOE) and construct the surrogate model
        """
        self.logger.info('selected surrogate model: {}'.format(self.surrogate.__class__)) 
        self.logger.info('building the initial design of experiemnts...')

        samples = self._space.sampling(self.n_init_sample)
        self.data = [Solution(s, index=k, var_name=self.var_names) for k, s in enumerate(samples)]
        self.evaluate(self.data, runs=self.init_n_eval)
        
        # set the initial incumbent
        fitness = np.array([s.fitness for s in self.data])

        self.incumbent_id = np.nonzero(fitness == self._best(fitness))[0][0]
        self.fit_and_assess()

        # record the incumbent in iteration 0
        # self.data.loc[[self.incumbent_id]].to_csv(self.data_file, header=True, index=False, mode='w')

    def step(self):
        if not hasattr(self, 'data'):
           self._initialize()
        
        ids = self.select_candidate()
        if self.noisy:
            self.incumbent_id = self.intensify(ids)
        else:
            fitness = np.array([s.fitness for s in self.data])
            self.incumbent_id = np.nonzero(fitness == self._best(fitness))[0][0]

        self.incumbent = self.data[self.incumbent_id]
        
        # model re-training
        # TODO: test more control rules on model refitting
        # if self.eval_count % 2 == 0:
            # self.fit_and_assess()
        self.fit_and_assess()
        self.iter_count += 1
        self.hist_f.append(self.incumbent.fitness)

        self.logger.info('iteration {}, current incumbent is:'.format(self.iter_count))
        self.logger.info(self.incumbent.to_dict())
        
        # save the iterative data configuration to csv
        # self.incumbent.to_csv(self.data_file, header=False, index=False, mode='a')
        return self.incumbent, self.incumbent.fitness

    def run(self):
        while not self.check_stop():
            self.step()

        self.stop_dict['n_eval'] = self.eval_count
        self.stop_dict['n_iter'] = self.iter_count
        return self.incumbent, self.stop_dict

    def check_stop(self):
        # TODO: add more stop criteria
        # unify the design purpose of stop_dict
        if self.iter_count >= self.max_iter:
            self.stop_dict['max_iter'] = True

        if self.eval_count >= self.max_eval:
            self.stop_dict['max_eval'] = True
        
        if self.ftarget is not None and hasattr(self, 'incumbent') and \
            self._compare(self.incumbent.perf, self.ftarget):
            self.stop_dict['ftarget'] = True

        return len(self.stop_dict)

    def _acquisition(self, plugin=None, dx=False):
        if plugin is None:
            # plugin = np.min(self.data.perf) if self.minimize else -np.max(self.data.perf)
            # Note that performance are normalized when building the surrogate
            plugin = 0 if self.minimize else -1
            
        if self.n_point > 1:  # multi-point method
            # create a portofolio of n infill-criteria by 
            # instantiating n 't' values from the log-normal distribution
            # exploration and exploitation
            # TODO: perhaps also introduce cooling schedule for MGF
            # TODO: other method: niching, UCB, q-EI
            tt = np.exp(0.5 * np.random.randn())
            acquisition_func = MGFI(self.surrogate, plugin, minimize=self.minimize, t=tt)
            
        elif self.n_point == 1: # sequential mode
            
            if self.infill == 'EI':
                acquisition_func = EI(self.surrogate, plugin, minimize=self.minimize)
            elif self.infill == 'PI':
                acquisition_func = PI(self.surrogate, plugin, minimize=self.minimize)
            elif self.infill == 'MGFI':
                acquisition_func = MGFI(self.surrogate, plugin, minimize=self.minimize, t=self.t)
                self._annealling()
            elif self.infill == 'UCB':
                raise NotImplementedError
                
        return functools.partial(acquisition_func, dx=dx)
        
    def _annealling(self):
        if self.schedule == 'exp':  
             self.t *= self.alpha
        elif self.schedule == 'linear':
            self.t -= self.eta
        elif self.schedule == 'log':
            # TODO: verify this
            self.t = self.c / np.log(self.iter_count + 1 + 1)
        
    def arg_max_acquisition(self, plugin=None):
        """
        Global Optimization on the acqusition function 
        """
        if self.verbose:
            self.logger.info('acquisition function optimziation...')
        
        dx = True if self._optimizer == 'BFGS' else False
        obj_func = [self._acquisition(plugin, dx=dx) for i in range(self.n_point)]

        if self.n_point == 1:
            candidates, values = self._argmax_multistart(obj_func[0])
        else:
            # parallelization using joblib
            res = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._argmax_multistart, check_pickle=False)(func) for func in obj_func)
            candidates, values = list(zip(*res))
        return candidates, values

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
                
                if stop_dict["warnflag"] != 0 and self.verbose:
                    self.logger.warn("L-BFGS-B terminated abnormally with the "
                                     " state: %s" % stop_dict)
                                
            elif self._optimizer == 'MIES':
                opt = mies(x0, obj_func, self._bounds.T, self._levels, self.param_type, 
                           eval_budget, minimize=False, verbose=False)                            
                xopt_, fopt_, stop_dict = opt.optimize()

            if fopt_ > best:
                best = fopt_
                wait_count = 0
                if self.verbose:
                    self.logger.info('restart : {} - funcalls : {} - Fopt : {}'.format(iteration + 1, 
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
        assert hasattr(self.obj_func, '__call__')

        if np.isinf(self.max_eval) and np.isinf(self.max_iter):
            raise ValueError('max_eval and max_iter cannot be both infinite')
