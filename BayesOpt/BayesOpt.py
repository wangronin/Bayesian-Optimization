# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:05:01 2017

@author: wangronin
"""
from __future__ import division
from __future__ import print_function

import pdb
import warnings, dill, functools, itertools, copy_reg
from joblib import Parallel, delayed

import pandas as pd
import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics import r2_score

from .InfillCriteria import EI, MGFI
from .optimizer import mies
from .utils import proportional_selection

# TODO: remove the usage of pandas here change it to customized np.ndarray
# TODO: adding logging system

class BayesOpt(object):
    """
    Generic Bayesian optimization algorithm
    """
    def __init__(self, search_space, obj_func, surrogate, 
                 minimize=True, noisy=False, eval_budget=None, max_iter=None, 
                 n_init_sample=None, n_point=1, n_jobs=1, backend='multiprocessing',
                 n_restart=None, optimizer='MIES', wait_iter=3,
                 verbose=False, random_seed=None):
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
            eval_budget : int,
                maximal number of evaluations on the objective function
            max_iter : int,
                maximal iteration
            n_init_sample : int,
                the size of inital Design of Experiment (DoE),
                default: 20 * dim
            n_point : int,
                the number of candidate solutions proposed using infill-criteria,
                default : 1
            n_jobs : int,
                the number of jobs scheduled for parallelizing the evaluation. 
                Only Effective when n_point > 1 
            backend : str, 
                the parallelization backend, supporting: 'multiprocessing', 'MPI', 'SPARC'
            optimizer: str,
                the optimization algorithm for infill-criteria,
                supported options: 'MIES' (Mixed-Integer Evolution Strategy for random forest), 
                                   'BFGS' (quasi-Newtion for GPR)
        """
        self.verbose = verbose
        self._space = search_space
        self.var_names = self._space.var_name.tolist()
        self.obj_func = obj_func
        self.noisy = noisy
        self.surrogate = surrogate
        self.n_point = n_point
        self.n_jobs = min(self.n_point, n_jobs)
        self._parallel_backend = backend

        self.minimize = minimize
        self.dim = len(self._space)

        # column names for each variable type
        self.con_ = self._space.var_name[self._space.id_C].tolist()   # continuous
        self.cat_ = self._space.var_name[self._space.id_N].tolist()   # categorical
        self.int_ = self._space.var_name[self._space.id_O].tolist()   # integer

        self.param_type = self._space.var_type
        self.N_r = len(self.con_)
        self.N_d = len(self.cat_)
        self.N_i = len(self.int_)
       
        # parameter: objective evaluation
        self.init_n_eval = 1      # TODO: for noisy objective function, maybe increase the initial evaluations
        self.max_eval = int(eval_budget) if eval_budget else np.inf
        self.max_iter = int(max_iter) if max_iter else np.inf
        self.n_init_sample = self.dim * 20 if n_init_sample is None else int(n_init_sample)
        self.eval_hist = []
        self.eval_hist_id = []
        self.iter_count = 0
        self.eval_count = 0

        # paramter: acquisition function optimziation
        mask = np.nonzero(self._space.C_mask | self._space.O_mask)[0]
        self._bounds = np.array([self._space.bounds[i] for i in mask])             # bounds for continuous and integer variable
        # self._levels = list(self._space.levels.values())
        self._levels = np.array([self._space.bounds[i] for i in self._space.id_N]) # levels for discrete variable
        self._optimizer = optimizer
        self._max_eval = int(5e2 * self.dim) 
        self._random_start = int(10 * self.dim) if n_restart is None else n_restart
        self._wait_iter = int(wait_iter)    # maximal restarts when optimal value does not change

        # Intensify: the number of potential configuations compared against the current best
        # self.mu = int(np.ceil(self.n_init_sample / 3))
        self.mu = 3
        
        # stop criteria
        self.stop_dict = {}
        self.hist_perf = []
        self._check_params()

        # set the random seed
        self.random_seed = random_seed
        if self.random_seed:
            np.random.seed(self.random_seed)
            
        copy_reg.pickle(self._eval, dill.pickles) # for pickling 

    def _get_var(self, data):
        """
        get variables from the dataframe
        """
        var_list = lambda row: [_ for _ in row[self.var_names].values]
        if isinstance(data, pd.DataFrame):
            return [var_list(row) for i, row in data.iterrows()]
        elif isinstance(data, pd.Series):
            return var_list(data)
    
    def _to_dataframe(self, var, index=0):
        if not hasattr(var[0], '__iter__'):
            var = [var]
        var = np.array(var, dtype=object)
        N = len(var)
        df = pd.DataFrame(np.c_[var, [0] * N, [None] * N],
                          columns=self.var_names + ['n_eval', 'perf'])
        df[self.con_] = df[self.con_].apply(pd.to_numeric)
        df[self.int_] = df[self.int_].apply(lambda c: pd.to_numeric(c, downcast='integer'))
        df.index = list(range(index, index + df.shape[0]))
        return df

    def _compare(self, perf1, perf2):
        if self.minimize:
            return perf1 < perf2
        else:
            return perf1 > perf2
    
    def _remove_duplicate(self, confs):
        """
        check for the duplicated solutions, as it is not allowed
        for noiseless objective functions
        """
        idx = []
        X = self.data[self.var_names]
        for i, x in confs.iterrows():
            x_ = pd.to_numeric(x[self.con_])
            CON = np.all(np.isclose(X[self.con_].values, x_), axis=1)
            INT = np.all(X[self.int_] == x[self.int_], axis=1)
            CAT = np.all(X[self.cat_] == x[self.cat_], axis=1)
            if not any(CON & INT & CAT):
                idx.append(i)
        return confs.loc[idx]

    def _eval(self, x, runs=1):
        perf_, n_eval = x.perf, x.n_eval
        # TODO: handle the input type in a better way
        try:    # for dictionary input
            __ = [self.obj_func(x[self.var_names].to_dict()) for i in range(runs)]
        except: # for list input
            __ = [self.obj_func(self._get_var(x)) for i in range(runs)]
        perf = np.sum(__)

        x.perf = perf / runs if not perf_ else np.mean((perf_ * n_eval + perf))
        x.n_eval += runs

        self.eval_count += runs
        self.eval_hist += __
        self.eval_hist_id += [x.name] * runs
        
        return x, runs, __, [x.name] * runs

    def evaluate(self, data, runs=1):
        """ Evaluate the candidate points and update evaluation info in the dataframe
        """
        if isinstance(data, pd.Series):
            self._eval(data)
        
        elif isinstance(data, pd.DataFrame): 
            if self.n_jobs > 1:
                if self._parallel_backend == 'multiprocessing': # parallel execution using joblib
                    res = Parallel(n_jobs=self.n_jobs, verbose=False)(
                        delayed(self._eval, check_pickle=False)(row) for k, row in data.iterrows())
                    
                    x, runs, hist, hist_id = zip(*res)
                    self.eval_count += sum(runs)
                    self.eval_hist += list(itertools.chain(*hist))
                    self.eval_hist_id += list(itertools.chain(*hist_id))
                    for i, k in enumerate(data.index):
                        data.loc[k] = x[i]
                elif self._parallel_backend == 'MPI': # parallel execution using MPI
                    # TODO: to use InstanceRunner here
                    pass
                elif self._parallel_backend == 'Spark': # parallel execution using Spark
                    pass        
            else:
                for k, row in data.iterrows():
                    self._eval(row)
                    data.loc[k, ['n_eval', 'perf']] = row[['n_eval', 'perf']]

    def fit_and_assess(self):
        X, perf = self._get_var(self.data), self.data['perf'].values

        # normalization the response for numerical stability
        # e.g., for MGF-based acquisition function
        perf_min = np.min(perf)
        perf_max = np.max(perf)
        perf_ = (perf - perf_min) / (perf_max - perf_min)

        # fit the surrogate model
        self.surrogate.fit(X, perf_)
        
        self.is_update = True
        perf_hat = self.surrogate.predict(X)
        self.r2 = r2_score(perf_, perf_hat)

        # TODO: in case r2 is really poor, re-fit the model or transform the input? 
        # consider the performance metric transformation in SMAC
        if self.verbose:
            print('Surrogate model r2: {}'.format(self.r2))
        return self.r2

    def select_candidate(self):
        self.is_update = False
        # always generate mu + 1 candidate solutions
        while True:
            confs_, acqui_opts_ = self.arg_max_acquisition()
            confs_ = self._to_dataframe(confs_, self.data.shape[0])
            confs_ = self._remove_duplicate(confs_)

            # if no new design site is found, re-estimate the parameters immediately
            if len(confs_) == 0:
                if not self.is_update:
                    # Duplication are commonly encountered in the 'corner'
                    self.fit_and_assess()
                else:
                    warnings.warn('iteration {}: duplicated solution found \
                                by optimization! New points is taken from random \
                                design'.format(self.iter_count))
                    confs_ = self.sampling(N=1)
                    break
            else:
                break

        candidates_id = list(confs_.index)
        if self.noisy:
            id_ = self.data[self.data.id != self.incumbent_id.id].id
            perf = self.data[self.data.id != self.incumbent_id.id].perf
            __ = proportional_selection(perf, self.mu, self.minimize, replacement=False)
            candidates_id.append(id_[__])
        
        # TODO: postpone the evaluate to intensify...
        self.evaluate(confs_, runs=self.init_n_eval)
        self.data = self.data.append(confs_)
        self.data.perf = pd.to_numeric(self.data.perf)
        return candidates_id

    def intensify(self, candidates_ids):
        """
        intensification procedure for noisy observations (from SMAC)
        """
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
        if self.verbose:
            print('selected surrogate model:', self.surrogate.__class__) 
            print('building the initial design of experiemnts...')

        # self.data = self.sampling(self.n_init_sample)
        samples = self._space.sampling(self.n_init_sample)
        self.data = self._to_dataframe(samples)
        self.evaluate(self.data, runs=self.init_n_eval)
        
        # set the initial incumbent
        self.data.perf = pd.to_numeric(self.data.perf)
        perf = np.array(self.data.perf)

        self.incumbent_id = np.nonzero(perf == np.min(perf))[0][0]
        self.fit_and_assess()

    def step(self):
        if not hasattr(self, 'data'):
           self._initialize()
        
        ids = self.select_candidate()
        if self.noisy:
            self.incumbent_id = self.intensify(ids)
        else:
            perf = np.array(self.data.perf)
            self.incumbent_id = np.nonzero(perf == np.min(perf))[0][0]

        # model re-training
        self.fit_and_assess()
        self.iter_count += 1
        self.hist_perf.append(self.data.loc[self.incumbent_id, 'perf'])
        
        if self.verbose:
            print()
            print('iteration {}, current incumbent is:'.format(self.iter_count))
            print(self.data.loc[[self.incumbent_id]])
            print()
        
        incumbent = self.data.loc[[self.incumbent_id]]
        return self._get_var(incumbent)[0], incumbent.perf.values

    def run(self):
        while not self.check_stop():
            self.step()

        self.stop_dict['n_eval'] = self.eval_count
        self.stop_dict['n_iter'] = self.iter_count
        incumbent = self.data.loc[[self.incumbent_id]]
        return incumbent, self.stop_dict

    def check_stop(self):
        # TODO: add more stop criteria
        if self.iter_count >= self.max_iter:
            self.stop_dict['max_iter'] = True

        if self.eval_count >= self.max_eval:
            self.stop_dict['max_eval'] = True

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
            t = np.exp(0.5 * np.random.randn())
            acquisition_func = MGFI(self.surrogate, plugin, minimize=self.minimize, t=t)
        elif self.n_point == 1:
            acquisition_func = EI(self.surrogate, plugin, minimize=self.minimize)
        return functools.partial(acquisition_func, dx=dx)
        
    def arg_max_acquisition(self, plugin=None):
        """
        Global Optimization on the acqusition function 
        """
        if self.verbose:
            print('acquisition function optimziation...')
        
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
            x0 = self._space.sampling(1)[0]
            
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
                fopt_ = -np.sum(fopt_)
                
                if stop_dict["warnflag"] != 0 and self.verbose:
                    warnings.warn("L-BFGS-B terminated abnormally with the "
                                  " state: %s" % stop_dict)
                                
            elif self._optimizer == 'MIES':
                opt = mies(x0, obj_func, self._bounds.T, self._levels, self.param_type, 
                           eval_budget, minimize=False, verbose=False)                            
                xopt_, fopt_, stop_dict = opt.optimize()

            if fopt_ > best:
                best = fopt_
                wait_count = 0
                if self.verbose:
                    print('[DEBUG] restart : {} - funcalls : {} - Fopt : {}'.format(iteration + 1, 
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
