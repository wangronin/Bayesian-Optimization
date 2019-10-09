# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:05:01 2017

@author: Hao Wang
@email: wangronin@gmail.com

"""
from pdb import set_trace
import os, sys, dill, functools, logging, time

sys.path.insert(0, os.path.abspath('../'))

import pandas as pd
import numpy as np

from utils import bcolors, MyFormatter

# TODO: replace 'pathos' in long runs as this package create global process pool 
# IMPORTANT: pathos is better than joblib 
# it uses dill for pickling, which allows for pickling functions
# from joblib import Parallel, delayed
from pathos.multiprocessing import ProcessingPool 
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

from .base import Solution
from .optimizer import mies, cma_es
from .InfillCriteria import EI, PI, MGFI
from .Surrogate import SurrogateAggregation
from .misc import proportional_selection, non_dominated_set_2d

# TODO: implement the automatic surrogate model selection
# TODO: improve the efficiency; profiling
class BO(object):
    """Bayesian Optimization (BO) base class"""
    def __init__(self, search_space, obj_func, surrogate, ftarget=None,
                 eq_func=None, ineq_func=None, minimize=True, max_eval=None, 
                 max_iter=None, init_points=None,
                 infill='EI', t0=2, tf=1e-1, schedule='exp', eval_type='list',
                 n_init_sample=None, n_point=1, n_job=1, backend='multiprocessing',
                 n_restart=None, max_infill_eval=None, wait_iter=3, optimizer='MIES', 
                 data_file=None, verbose=False, random_seed=None):
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
            backend : str, 
                the parallelization backend, supporting: 'multiprocessing', 'MPI', 'SPARC'
            optimizer: str,
                the optimization algorithm for infill-criteria,
                supported options are:
                    'MIES' (Mixed-Integer Evolution Strategy), 
                    'BFGS' (quasi-Newtion for GPR)
        """
        self.verbose = verbose
        self.data_file = data_file
        self._space = search_space
        self.var_names = self._space.var_name
        self.obj_func = obj_func
        self.eq_func = eq_func
        self.ineq_func = ineq_func
        self.surrogate = surrogate
        self.n_point = int(n_point)
        self.n_job = int(n_job)
        self._parallel_backend = backend
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
        self.set_logger(None)

        # setup multi-processing workers
        if self.n_job > 1:
            self.pool = ProcessingPool(ncpus=self.n_job)

    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.terminate()

    def set_logger(self, logger):
        """Create the logging object
        Params:
            logger : str, None or logging.Logger,
                either a logger file name, None (no logging) or a logger object
        """
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

    def _compare(self, f1, f2):
        """Test if objecctive value f1 is better than f2
        """
        return f1 < f2 if self.minimize else f2 > f1
    
    def _remove_duplicate(self, data):
        """
        check for the duplicated solutions, as it is not allowed
        for noiseless objective functions
        """
        _ = []
        for i in range(data.N):
            x = data[i]
            
            CON = np.all(np.isclose(np.asarray(self.data[:, self.r_index], dtype='float'),
                                    np.asarray(x[self.r_index], dtype='float')), axis=1)
            INT = np.all(self.data[:, self.i_index] == x[self.i_index], axis=1)
            CAT = np.all(self.data[:, self.d_index] == x[self.d_index], axis=1)
            if not any(CON & INT & CAT):
                _ += [i]
        return data[_]
        
    def evaluate(self, data, runs=1):
        """Evaluate the candidate points and update evaluation info in the dataframe
        """
        data = np.atleast_2d(data)
        N = len(data)

        if self.n_job > 1:
            if self._eval_type == 'list':
                X = [x.tolist() for x in data]
            elif self._eval_type == 'dict':
                X = [self._space.to_dict(x) for x in data]

            ans = np.zeros((runs, N))
            for i in range(runs):
                # parallelization is implemented in self.obj_func
                ans[i, :] = self.obj_func(X, n_jobs=self.n_job)
            
            ans = np.atleast_2d(ans)
            ans = np.mean(ans, axis=0)
            
            for i, k in enumerate(data):
                data[i].fitness = ans[i]
                data[i].n_eval += runs

        else:  # sequential execution
            for x in data:
                if self._eval_type == 'list':
                    ans = [self.obj_func(x.tolist()) for i in range(runs)]
                elif self._eval_type == 'dict':
                    ans = [self.obj_func(self._space.to_dict(x)) for i in range(runs)]

                x.fitness = np.mean(np.asarray(ans))
                x.n_eval += runs

        self.eval_count += runs * N

    def fit_and_assess(self):
        fitness = self.data.fitness
        # normalization the response for the numerical stability
        # e.g., for MGF-based acquisition function
        self.fmin, self.fmax = np.min(fitness), np.max(fitness)

        # flat_fitness = np.isclose(self.fmin, self.fmax)
        fitness_scaled = (fitness - self.fmin) / (self.fmax - self.fmin)
        self.frange = self.fmax - self.fmin

        # fit the surrogate model
        self.surrogate.fit(self.data, fitness_scaled)
        
        self.is_update = True
        fitness_hat = self.surrogate.predict(self.data)
        r2 = r2_score(fitness_scaled, fitness_hat)

        # TODO: adding cross validation for the model? 
        # TODO: how to prevent overfitting in this case
        # TODO: in case r2 is really poor, re-fit the model or transform the input? 
        # TODO: perform diagnostic/validation on the surrogate model
        # consider the performance metric transformation in SMAC
        self.logger.info('Surrogate model r2: {}'.format(r2))
        return r2

    def select_candidate(self):
        self.is_update = False
        X, infill_value = self.arg_max_acquisition()
        X = Solution(X, index=len(self.data) + np.arange(len(X)), var_name=self.var_names)
        
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
            X = X.tolist() + s 
            X = Solution(X, index=len(self.data) + np.arange(len(X)), var_name=self.var_names)
        
        return X
    
    def _initialize(self):
        """Generate the initial data set (DOE) and construct the surrogate model"""
        if hasattr(self, 'data'):
            self.logger.warn('initialization is already performed!') 
            return 

        self.logger.info('selected surrogate model: {}'.format(self.surrogate.__class__)) 
        self.logger.info('building {:d} initial design points...'.format(self.n_init_sample))
        
        sampling_trial = self._init_flatfitness_trial
        while True:
            if self.init_points is not None:
                n = len(self.init_points)
                DOE = self.init_points + self._space.sampling(self.n_init_sample - n)
            else:
                DOE = self._space.sampling(self.n_init_sample)
                
            DOE = Solution(DOE, var_name=self.var_names, n_obj=self.n_obj)
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
            self.logger.info('DOE {}, fitness: {} -- {}'.format(i + 1, x.fitness, self._space.to_dict(x)))

        self.fit_and_assess()
        if self.data_file is not None: # save the initial design to csv
            self.data.to_csv(self.data_file)

    def after_eval_check(self, X):
        _ = np.isnan(X.fitness)
        if np.any(_):
            if len(_.shape) == 2:
                _ = np.any(_, axis=1).ravel()
            self.logger.warn('{} candidate solutions are removed '
                             'due to falied fitness evaluation: \n{}'.format(sum(_), str(X[_, :])))
            X = X[~_, :] 
        return X

    def step(self):
        X = self.select_candidate()     # mutation by optimization

        t0 = time.time()
        self.evaluate(X, runs=self.init_n_eval)
        self.logger.info('evaluation takes {:.4f}s'.format(time.time() - t0))
        
        X = self.after_eval_check(X)
        self.data = self.data + X

        if self.data_file is not None:
            X.to_csv(self.data_file, header=False, append=True)

        self.fopt = self._best(self.data.fitness)
        _ = np.nonzero(self.data.fitness == self.fopt)[0][0]
        self.xopt = self.data[_]   

        self.logger.info(bcolors.WARNING + \
            'iteration {}, objective value: {:.8f}'.format(self.iter_count, 
            self.xopt.fitness[0]) + bcolors.ENDC)
        self.logger.info('xopt: {}'.format(self._space.to_dict(self.xopt)))     
        
        self.fit_and_assess()     # re-train the surrogate model   
        self.iter_count += 1
        self.hist_f.append(self.xopt.fitness)
        
        return self.xopt.tolist(), self.xopt.fitness

    def run(self):
        self._initialize() 
        while not self.check_stop():
            self.step()

        self.stop_dict['n_eval'] = self.eval_count
        self.stop_dict['n_iter'] = self.iter_count
        self.logger.handlers = []   # completely de-register the logger

        return self.xopt.tolist(), self.xopt.fitness, self.stop_dict

    def check_stop(self):
        # TODO: add more stop criteria
        if self.iter_count >= self.max_iter:
            self.stop_dict['max_iter'] = True

        if self.eval_count >= self.max_eval:
            self.stop_dict['max_eval'] = True
        
        if self.ftarget is not None and hasattr(self, 'xopt'):
            if self._compare(self.xopt.fitness, self.ftarget):
                self.stop_dict['ftarget'] = True

        return len(self.stop_dict)

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
        
        dx = True if self._optimizer == 'BFGS' else False
        criteria = [self._acquisition(plugin, dx=dx) for i in range(self.n_point)]
        
        if self.n_job > 1:
            # TODO: fix this issue once for all!
            try:  
                self.pool.restart() # restart the pool in case it is terminated before
            except AssertionError:
                pass
            __ = self.pool.map(self._argmax_multistart, [_ for _ in criteria])
            # __ = Parallel(n_jobs=self.n_job)(delayed(self._argmax_multistart)(c) for c in criteria)
        else:
            __ = [list(self._argmax_multistart(_)) for _ in criteria]

        candidates, values = tuple(zip(*__))
        self.logger.debug('infill criteria optimziation takes {:.4f}s'.format(time.time() - t0))

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
                           max_eval=eval_budget, minimize=False, verbose=False)                           
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
        # assert hasattr(self.obj_func, '__call__')
        if np.isinf(self.max_eval) and np.isinf(self.max_iter):
            raise ValueError('max_eval and max_iter cannot be both infinite')


# TODO: validate this subclass
# TODO: move those to Extension.py
class BOAnnealing(BO):
    def __init__(self, t0, tf, schedule, *argv, **kwargs):
        super(BOAnnealing, self).__init__(*argv, **kwargs)
        assert self.infill in ['MGFI', 'UCB']
        self.t0 = t0
        self.tf = tf
        self.t = t0
        self.schedule = schedule
            
        max_iter = self.max_eval - self.n_init_sample
        if self.schedule == 'exp':                         # exponential
            self.alpha = (self.tf / t0) ** (1. / max_iter) 
        elif self.schedule == 'linear':
            self.eta = (t0 - self.tf) / max_iter           # linear
        elif self.schedule == 'log':
            self.c = self.tf * np.log(max_iter + 1)        # logarithmic 

    def _annealling(self):
        if self.schedule == 'exp':  
             self.t *= self.alpha
        elif self.schedule == 'linear':
            self.t -= self.eta
        elif self.schedule == 'log':
            # TODO: verify this
            self.t = self.c / np.log(self.iter_count + 1 + 1)
    
    def _acquisition(self, plugin=None, dx=False):
        """
        plugin : float,
            the minimal objective value used in improvement-based infill criteria
            Note that it should be given in the original scale
        """
        infill = super(BOAnnealing, self)._acquisition(plugin, dx)
        if self.n_point == 1 and self.infill == 'MGFI':
            self._annealling()
                
        return infill


class BOAdapt(BO):
    def __init__(self, *argv, **kwargs):
        super(BOAdapt, self).__init__(*argv, **kwargs)


class BONoisy(BO):
    def __init__(self, *argv, **kwargs):
        super(BONoisy, self).__init__(*argv, **kwargs)
        self.noisy = True
        self.infill = 'EQI'
        # Intensify: the number of potential configuations compared against the current best
        self.mu = 3
    
    def step(self):
        self._initialize()  # initialization
        
        # TODO: postpone the evaluate to intensify...
        X = self.select_candidate() 
        self.evaluate(X, runs=self.init_n_eval)
        self.data += X

        # for noisy fitness: perform a proportional selection from the evaluated ones
        id_, fitness = zip([(i, d.fitness) for i, d in enumerate(self.data) \
                            if i != self.incumbent_id])
        # __ = proportional_selection(fitness, self.mu, self.minimize, replacement=False)
        # candidates_id.append(id_[__])
        
        # self.incumbent_id = self.intensify(ids)
        self.incumbent = self.data[self.incumbent_id]
        
        # TODO: implement more control rules for model refitting
        self.fit_and_assess()
        self.iter_count += 1
        self.hist_f.append(self.incumbent.fitness)

        self.logger.info(bcolors.WARNING + \
            'iteration {}, objective value: {}'.format(self.iter_count, 
            self.incumbent.fitness) + bcolors.ENDC)
        self.logger.info('incumbent: {}'.format(self.incumbent.to_dict()))

        # save the incumbent to csv
        incumbent_df = pd.DataFrame(np.r_[self.incumbent, self.incumbent.fitness].reshape(1, -1))
        incumbent_df.to_csv(self.data_file, header=False, index=False, mode='a')
        
        return self.incumbent, self.incumbent.fitness
            
    def intensify(self, candidates_ids):
        """
        intensification procedure for noisy observations (from SMAC)
        """
        # TODO: verify the implementation here
        # maxR = 20 # maximal number of the evaluations on the incumbent
        # for i, ID in enumerate(candidates_ids):
        #     r, extra_run = 1, 1
        #     conf = self.data.loc[i]
        #     self.evaluate(conf, 1)
        #     print(conf.to_frame().T)

        #     if conf.n_eval > self.incumbent_id.n_eval:
        #         self.incumbent_id = self.evaluate(self.incumbent_id, 1)
        #         extra_run = 0

        #     while True:
        #         if self._compare(self.incumbent_id.perf, conf.perf):
        #             self.incumbent_id = self.evaluate(self.incumbent_id, 
        #                                            min(extra_run, maxR - self.incumbent_id.n_eval))
        #             print(self.incumbent_id.to_frame().T)
        #             break
        #         if conf.n_eval > self.incumbent_id.n_eval:
        #             self.incumbent_id = conf
        #             if self.verbose:
        #                 print('[DEBUG] iteration %d -- new incumbent selected:' % self.iter_count)
        #                 print('[DEBUG] {}'.format(self.incumbent_id))
        #                 print('[DEBUG] with performance: {}'.format(self.incumbent_id.perf))
        #                 print()
        #             break

        #         r = min(2 * r, self.incumbent_id.n_eval - conf.n_eval)
        #         self.data.loc[i] = self.evaluate(conf, r)
        #         print(self.conf.to_frame().T)
        #         extra_run += r

# TODO: 
class SMS_BO(BO):
    pass


class MOBO_D(BO):
    """Decomposition-based Multi-Objective Bayesian Optimization (MO-EGO/D) 
    """
    # TODO: this number should be set according to the capability of the server
    # TODO: implement Tchebycheff scalarization
    __max_procs__ = 16  # maximal number of processes

    def _eval(x, _eval_type, obj_func, _space=None, logger=None, runs=1, pickling=False):
        """evaluate one solution

        Parameters
        ----------
        x : bytes,
            serialization of the Solution instance
        """
        if pickling:
            # TODO: move the pickling/unpickling operation to class 'Solution'
            x = dill.loads(x)
        fitness_, n_eval = x.fitness.flatten(), x.n_eval

        if hasattr(obj_func, '__call__'):   # vector-valued obj_func
            if _eval_type == 'list':
                ans = [obj_func(x.tolist()) for i in range(runs)]
            elif _eval_type == 'dict':
                ans = [obj_func(_space.to_dict(x)) for i in range(runs)]

            # TODO: this should be done per objective fct.
            fitness = np.sum(np.asarray(ans), axis=0)

            # TODO: fix it
            # x.fitness = fitness / runs if any(np.isnan(fitness_)) \
            #     else (fitness_ * n_eval + fitness) / (x.n_eval + runs)
            x.fitness = fitness / runs 

        elif hasattr(obj_func, '__iter__'):  # a list of  obj_func
            for i, obj_func in enumerate(obj_func):
                try:
                    if _eval_type == 'list':
                        ans = [obj_func(x.tolist()) for i in range(runs)]
                    elif _eval_type == 'dict':
                        ans = [obj_func(_space.to_dict(x)) for i in range(runs)]
                except Exception as ex:
                    logger.error('Error in function evaluation: {}'.format(ex))
                    return

                fitness = np.sum(ans)
                x.fitness[0, i] = fitness / runs if np.isnan(fitness_[i]) \
                    else (fitness_[i] * n_eval + fitness) / (x.n_eval + runs)

        x.n_eval += runs
        return dill.dumps(x)
    
    def __init__(self, n_obj=2, aggregation='WS', n_point=5, n_job=1, *argv, **kwargs):
        """
        Arguments
        ---------
        n_point : int,
            the number of evaluated points in each iteration
        aggregation: str or callable,
            the scalarization method/function. Supported options are:
                'WS' : weighted sum
                'Tchebycheff' : Tchebycheff scalarization
        """
        super(MOBO_D, self).__init__(*argv, **kwargs)
        self.n_point = int(n_point)
        # TODO: perhaps leave this an input parameter
        self.mu = 2 * self.n_point   # the number of generated points
        self.n_obj = int(n_obj)
        assert self.n_obj > 1

        if isinstance(self.minimize, bool):
            self.minimize = [self.minimize] * self.n_obj
        elif hasattr(self.minimize, '__iter__'):
            assert len(self.minimize) == self.n_obj

        self.minimize = np.asarray(self.minimize)

        if hasattr(self.obj_func, '__iter__'):
            assert self.n_obj == len(self.obj_func)
        
        assert self.n_obj == len(self.surrogate)
        self.n_job = min(MOBO_D.__max_procs__, self.mu, n_job)

        # TODO: implement the Tchebycheff approach
        if isinstance(aggregation, str):
            assert aggregation in ['WS', 'Tchebycheff']
        else:
            assert hasattr(aggregation, '__call__')
        self.aggregation = aggregation

        # generate weights
        self.weights = np.random.rand(self.mu, self.n_obj)
        self.weights /= np.sum(self.weights, axis=1).reshape(self.mu, 1)
        self.labels_ = KMeans(n_clusters=self.n_point).fit(self.weights).labels_
        self.frange = np.zeros(self.n_obj)

    def evaluate(self, data, runs=1):
        """Evaluate the candidate points and update evaluation info in the dataframe
        """
        _eval_fun = functools.partial(MOBO_D._eval, _eval_type=self._eval_type, _space=self._space,
                                      obj_func=self.obj_func, logger=self.logger, runs=runs,
                                      pickling=self.n_job > 1)
        if len(data.shape) == 1:
            _eval_fun(data)
        else: 
            if self.n_job > 1:
                 # parallel execution using multiprocessing
                if self._parallel_backend == 'multiprocessing':
                    data_pickle = [dill.dumps(d) for d in data]
                    __ = self.pool.map(_eval_fun, data_pickle)

                    x = [dill.loads(_) for _ in __]
                    self.eval_count += runs * len(data)
                    for i, k in enumerate(data):
                        data[i].fitness = x[i].fitness
                        data[i].n_eval = x[i].n_eval
            else:
                for x in data:
                    _eval_fun(x)

    def fit_and_assess(self):
        def _fit(surrogate, X, y):
            surrogate.fit(X, y)
            y_hat = surrogate.predict(X)
            r2 = r2_score(y, y_hat)    
            return surrogate, r2
        
        # NOTE: convert the fitness to minimization problem
        # objective values that are subject to maximization is revert to mimization 
        self.y = self.data.fitness.copy()
        self.y *= np.asarray([-1] * self.data.N).reshape(-1, 1) ** (~self.minimize)

        ymin, ymax = np.min(self.y), np.max(self.y)
        if np.isclose(ymin, ymax): 
            raise Exception('flat objective value!')

        # self._y: normalized objective values
        self._y = (self.y - ymin) / (ymax - ymin)

        # fit the surrogate models
        if self.n_job > 1 and 11 < 2:  
            __ = self.pool.map(_fit, *zip(*[(self.surrogate[i], self.data, 
                            self._y[:, i]) for i in range(self.n_obj)]))
        else:               
            __ = []
            for i in range(self.n_obj):
                __.append(list(_fit(self.surrogate[i], self.data, self._y[:, i])))

        self.surrogate, r2 = tuple(zip(*__))
        for i in range(self.n_obj):
            self.logger.info('F{} Surrogate model r2: {}'.format(i + 1, r2[i]))

    def step(self):
        # self._initialize()  
        
        X = self.select_candidate() 
        self.evaluate(X, runs=self.init_n_eval)
        self.data += X
        
        self.fit_and_assess()     
        self.iter_count += 1

        # TODO: implement a faster algorithm to detect non-dominated point only!
        # non-dominated sorting: self.y takes minimization issue into account
        nd_idx = non_dominated_set_2d(self.y)

        # xopt is the set of the non-dominated point now
        self.xopt = self.data[nd_idx]  
        self.logger.info('{}iteration {}, {} points in the Pareto front: {}\n{}'.format(bcolors.WARNING, 
            self.iter_count, len(self.xopt), bcolors.ENDC, str(self.xopt)))

        if self.data_file is not None:
            self.xopt.to_csv(self.data_file)

        return self.xopt, self.xopt.fitness
    
    def select_candidate(self):
        _ = self.arg_max_acquisition()
        X, value = np.asarray(_[0], dtype='object'), np.asarray(_[1])
        
        X_ = []
        # select the best point from each cluster
        # NOTE: "the best point" means the maximum of infill criteria 
        for i in range(self.n_point):
            v = value[self.labels_ == i]
            idx = np.nonzero(v == np.max(v))[0][0]
            X_.append(X[self.labels_ == i][idx].tolist())

        X = Solution(X_, index=len(self.data) + np.arange(len(X_)), 
                     var_name=self.var_names, n_obj=self.n_obj)
        X = self._remove_duplicate(X)

        # if the number of new design sites obtained is less than required,
        # draw the remaining ones randomly
        if len(X) < self.n_point:
            self.logger.warn("iteration {}: duplicated solution found " 
                             "by optimization! New points is taken from random "
                             "design".format(self.iter_count))
            N = self.n_point - len(X)
            s = self._space.sampling(N, method='uniform')
            X = Solution(X.tolist() + s, index=len(self.data) + np.arange(self.n_point),
                         var_name=self.var_names, n_obj=self.n_obj)

        return X

    def _acquisition(self, surrogate=None, plugin=None, dx=False):
        """Generate Infill Criteria based on surrogate models

        Parameters
        ----------
        surrogate : class instance
            trained surrogate model
        plugin : float,
            the minimal objective value used in improvement-based infill criteria
            Note that it should be given in the original scale
        """
        # objective values are normalized
        if plugin is None:
            plugin = 0

        # NOTE: the following 'minimize' parameter is set to always 'True'
        # as 
        if self.infill == 'EI':
            acquisition_func = EI(surrogate, plugin, minimize=True)
        elif self.infill == 'PI':
            acquisition_func = PI(surrogate, plugin, minimize=True)
        elif self.infill == 'MGFI':
            acquisition_func = MGFI(surrogate, plugin, minimize=True, t=self.t)
        elif self.infill == 'UCB':
            raise NotImplementedError
                
        return functools.partial(acquisition_func, dx=dx) 

    # TODO: implement evolutionary algorithms, e.g., MOEA/D to optimize of all subproblems simultaneously
    def arg_max_acquisition(self, plugin=None):
        """Global Optimization of the acqusition function / Infill criterion
        Arguments
        ---------
        plugin : float,
            the cut-off value for improvement-based criteria
            it is set to the current minimal target value

        Returns
        -------
            candidates: tuple of list,
                candidate solution (in list)
            values: tuple of float,
                criterion value of the candidate solution
        """
        self.logger.debug('infill criteria optimziation...')
        
        dx = True if self._optimizer == 'BFGS' else False
        surrogates = (SurrogateAggregation(self.surrogate, weights=w) for w in self.weights)
        gmin = [np.min(self._y.dot(w)) for w in self.weights]
        criteria = (self._acquisition(s, gmin[i], dx=dx) for i, s in enumerate(surrogates))

        if self.n_job > 1:
            __ = self.pool.map(self._argmax_multistart, [_ for _ in criteria])
        else:
            __ = [list(self._argmax_multistart(_)) for _ in criteria]

        candidates, values = tuple(zip(*__))
        return candidates, values


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
        def fitness(x):
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

        opt = BO(space, fitness, model, max_eval=300, verbose=True, n_job=1, n_point=3,
                 n_init_sample=3,
                 init_points=[[0, 0, 10, 'OK']])
        xopt, fopt, stop_dict = opt.run()

    if 11 < 2:
        def fitness0(x):
            x = np.asarray(x)
            return sum(x ** 2.)
        
        def fitness1(x):
            x = np.asarray(x)
            return -sum((x + 2) ** 2.)

        space = ContinuousSpace([-5, 5]) * 2
        model = (RandomForest(levels=None), RandomForest(levels=None))

        obj_func = lambda x: [fitness0(x), fitness1(x)]
        opt = MOBO_D(n_obj=2, search_space=space, obj_func=obj_func, 
                     n_point=5, n_job=16, n_init_sample=10,
                     minimize=[True, False],
                     surrogate=model, max_iter=100, verbose=True)
        xopt, fopt, stop_dict = opt.run()