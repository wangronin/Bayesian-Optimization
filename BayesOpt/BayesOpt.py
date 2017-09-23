# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:05:01 2017

@author: wangronin
"""

from __future__ import division   # important! for the float division

import pdb

import random, warnings

import pandas as pd
import numpy as np

from numpy.random import randint, rand
from scipy.optimize import fmin_l_bfgs_b

from criteria import EI
from MIES import MIES
# from cma_es import cma_es

from sklearn.metrics import r2_score

MACHINE_EPSILON = np.finfo(np.double).eps

class BayesOpt(object):
    """
    Generic Bayesian optimization algorithm
    """
    def __init__(self, search_space, obj_func, surrogate, 
                 eval_budget=None, max_iter=None, n_init_sample=None, 
                 minimize=True, noisy=False, wait_iter=3, 
                 n_restart=None, optimizer='MIES', 
                 verbose=False, random_seed=None,  debug=False):

        self.debug = debug
        self.verbose = verbose
        self._space = search_space
        self.var_names = search_space.var_name
        self.obj_func = obj_func
        self.noisy = noisy
        self.surrogate = surrogate

        self.minimize = minimize
        self.dim = len(self._space)

        # TODO: those should be move to the search space class
        self.con_ = self._space.get_continous() # continuous
        self.cat_ = self._space.get_norminal()  # nominal
        self.int_ = self._space.get_ordinal()  # nominal
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
        self._optimizer = optimizer
        self._max_eval = int(5e2 * self.dim) 
        self._random_start = int(10 * self.dim) if n_restart is None else n_restart
        self._wait_iter = int(wait_iter)    # maximal restarts when optimal value does not change
        mask = np.nonzero(self._space.C_mask | self._space.O_mask)[0]
        self._bounds = np.array([self._space.bounds[i] for i in mask])
        self._levels = self._space.get_levels()

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
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

    def _get_var(self, data):
        """
        get variables from the dataframe
        """
        var_list = lambda row: [_ for _ in row[self.var_names].values]
        if isinstance(data, pd.DataFrame):
            return [var_list(row) for i, row in data.iterrows()]
        elif isinstance(data, pd.Series):
            return var_list(data)

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
        return confs.iloc[idx, :]

    def evaluate(self, conf, runs=1):
        perf_, n_eval = conf.perf, conf.n_eval
        # TODO: handle the evaluation in a better way
        try:    # for dictionary input
            __ = [self.obj_func(conf[self.var_names].to_dict()) for i in range(runs)]
        except: # for list input
            __ = [self.obj_func(self._get_var(conf)) for i in range(runs)]
        perf = np.sum(__)

        conf.perf = perf / runs if not perf_ else np.mean((perf_ * n_eval + perf))
        conf.n_eval += runs

        self.eval_count += runs
        self.eval_hist += __
        self.eval_hist_id += [conf.index] * runs
        return conf

    def fit_and_assess(self):
        # fit the surrogate model
        X, perf = self._get_var(self.data), self.data['perf'].values
        self.surrogate.fit(X, perf)
        
        self.is_updated = True
        perf_hat = self.surrogate.predict(X)
        r2 = r2_score(perf, perf_hat)
        
        # TODO: in case r2 is really poor, re-fit the model or transform the input? 
        if self.verbose:
            print 'Surrogate model r2: {}'.format(r2)
        return r2

    def sampling(self, N):
        data = self._space.sampling(N)
        data = pd.DataFrame(np.c_[data, [0] * N, [None] * N], 
                            columns=self.var_names + ['n_eval', 'perf'])

        data[self.con_] = data[self.con_].apply(pd.to_numeric)
        data[self.int_] = data[self.int_].apply(pd.to_numeric)

        return data

    def select_candidate(self):
        self.is_updated = False
        # always generate mu + 1 candidate solutions
        while True:
            confs_, acqui_opts_ = self.arg_max_acquisition()
            if 1 < 2: # only use the best acquisition point 
                confs_ = [confs_[0]]
            N = self.data.shape[0]
            confs_ = pd.DataFrame([conf + [0, None] for i, conf in enumerate(confs_)],
                                   columns=self.var_names + ['n_eval', 'perf'])
            confs_ = self._remove_duplicate(confs_)
            confs_.index = range(N, N + confs_.shape[0])

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

        # proportional selection without replacement
        candidates_id = list(confs_.index)
        if self.noisy:
            id_curr = self.data[self.data.id != self.incumbent.id].id
            perf = self.data[self.data.id != self.incumbent.id].perf
            if self.minimize:
                perf = -perf
                perf -= np.min(perf)

            idx = np.argsort(perf)
            perf = perf[idx]
            id_curr = id_curr[idx]

            for i in range(self.mu):
                min_ = np.min(perf)
                prob = np.cumsum((perf - min_) / (np.sum(perf) - min_ * len(perf)))
                _ = np.nonzero(np.random.rand() <= prob)[0][0]

                candidates_id.append(id_curr[_])
                perf = np.delete(perf, _)
                id_curr = np.delete(id_curr, _)

        for i, conf in confs_.iterrows():
            confs_.loc[i] = self.evaluate(conf, runs=self.init_n_eval)
        self.data = self.data.append(confs_)
        self.data.perf = pd.to_numeric(self.data.perf)
        return candidates_id

    def intensify(self, candidates_ids):
        """
        intensification procedure for noisy observations (SMAC)
        """
        maxR = 20 # maximal number of the evaluations on the incumbent
        for i, ID in enumerate(candidates_ids):
            r, extra_run = 1, 1
            conf = self.data.loc[i]
            self.evaluate(conf, 1)
            print conf.to_frame().T

            if conf.n_eval > self.incumbent.n_eval:
                self.incumbent = self.evaluate(self.incumbent, 1)
                extra_run = 0

            while True:
                if self._compare(self.incumbent.perf, conf.perf):
                    self.incumbent = self.evaluate(self.incumbent, 
                                                   min(extra_run, maxR - self.incumbent.n_eval))
                    print self.incumbent.to_frame().T
                    break
                if conf.n_eval > self.incumbent.n_eval:
                    self.incumbent = conf
                    if self.verbose:
                        print '[DEBUG] iteration %d -- new incumbent selected:' % self.iter_count
                        print '[DEBUG] {}'.format(self.incumbent)
                        print '[DEBUG] with performance: {}'.format(self.incumbent.perf)
                        print
                    break

                r = min(2 * r, self.incumbent.n_eval - conf.n_eval)
                self.data.loc[i] = self.evaluate(conf, r)
                print self.conf.to_frame().T
                extra_run += r
    
    def _initialize(self):
        if self.verbose:
            print 'selected surrogate model:', self.surrogate.__class__ 
            print 'building the initial design of experiemnts...'

        self.data = self.sampling(self.n_init_sample)
        for i, conf in self.data.iterrows():
            self.data.loc[i] = self.evaluate(conf, runs=self.init_n_eval)
        
        # set the initial incumbent
        self.data.perf = pd.to_numeric(self.data.perf)
        perf = np.array(self.data.perf)
        self.incumbent = np.nonzero(perf == np.min(perf))[0][0]
        self.fit_and_assess()

    def step(self):
        if not hasattr(self, 'data'):
           self._initialize()
        
        ids = self.select_candidate()
        if self.noisy:
            self.incumbent = self.intensify(ids)
        else:
            perf = np.array(self.data.perf)
            self.incumbent = np.nonzero(perf == np.min(perf))[0][0]

        # model re-training
        self.fit_and_assess()
        self.iter_count += 1
        self.hist_perf.append(self.data.loc[self.incumbent, 'perf'])
        
        # only for debug purpose
        if self.debug:
            tmp = np.array([_ for _ in self.data.iloc[-1, 0:2].values])
            np.set_printoptions(precision=30)
            print self.iter_count, tmp, np.random.get_state()[2]
            
        if self.verbose:
            print 'iteration {}, current incumbent is:'.format(self.iter_count)
            print self.data.loc[[self.incumbent]]
            print 

    def run(self):
        while not self.check_stop():
            self.step()

        self.stop_dict['n_eval'] = self.eval_count
        self.stop_dict['n_iter'] = self.iter_count
        return self.incumbent, self.stop_dict

    def check_stop(self):
        # TODO: add more stop criteria
        if self.iter_count >= self.max_iter:
            self.stop_dict['max_iter'] = True

        if self.eval_count >= self.max_eval:
            self.stop_dict['max_eval'] = True

        return len(self.stop_dict)

    def _acquisition_func(self, plugin=None, dx=False):
        if plugin is None:
            plugin = np.min(self.data.perf) if self.minimize else np.max(self.data.perf)
        
        # TODO: add other criteria as options
        acquisition_func = EI(self.surrogate, plugin, minimize=self.minimize)
        def func(x):
            res = acquisition_func(x, dx=dx)
            return (-res[0], -res[1]) if dx else -res
        return func
        
    def arg_max_acquisition(self, plugin=None):
        """
        Global Optimization on the acqusition function 
        """
        eval_budget = self._max_eval
        fopt = np.inf
        optima, foptima = [], []
        wait_count = 0

        if self.verbose:
            print 'acquisition function optimziation...'
        
        # TODO: add IPOP-CMA-ES here for testing
        for iteration in range(self._random_start):
            x0 = self._get_var(self.sampling(1))[0]
            
            # TODO: when the surrogate is GP, implement a GA-BFGS hybrid algorithm
            if self._optimizer == 'BFGS':
                obj_func = self._acquisition_func(plugin, dx=True)
                xopt_, fopt_, stop_dict = fmin_l_bfgs_b(obj_func, x0, pgtol=1e-8,
                                                        factr=1e6, bounds=self._bounds,
                                                        maxfun=eval_budget)
                xopt_ = xopt_.flatten().tolist()
                fopt_ = fopt_.sum()
                
                if stop_dict["warnflag"] != 0 and self.verbose:
                    warnings.warn("L-BFGS-B terminated abnormally with the "
                                  " state: %s" % stop_dict)
                                  
            elif self._optimizer == 'MIES':
                obj_func = self._acquisition_func(plugin, dx=False)
                mies = MIES(obj_func, x0, self._bounds.T, self._levels,
                            self.param_type, eval_budget, minimize=True, 
                            verbose=False)                            
                xopt_, fopt_, stop_dict = mies.optimize()

            if fopt_ < fopt:
                fopt = fopt_
                wait_count = 0
                if self.verbose:
                    print '[DEBUG] restart : {} - funcalls : {} - Fopt : {}'.format(iteration + 1, 
                        stop_dict['funcalls'], fopt_)
            else:
                wait_count += 1

            eval_budget -= stop_dict['funcalls']
            optima.append(xopt_)
            foptima.append(-fopt_)
            
            if eval_budget <= 0 or wait_count >= self._wait_iter:
                break
        
        # sort the optima in descending order
        idx = np.argsort(foptima)[::-1]
        optima = [optima[_] for _ in idx]
        foptima = [foptima[_] for _ in idx]
            
        return optima, foptima

    def _check_params(self):
        assert hasattr(self.obj_func, '__call__')

        if np.isinf(self.max_eval) and np.isinf(self.max_iter):
            raise ValueError('max_eval and max_iter cannot be both infinite')

