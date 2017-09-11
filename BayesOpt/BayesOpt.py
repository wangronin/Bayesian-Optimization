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

from GaussianProcess import GaussianProcess_extra as GaussianProcess
from criteria import EI
from MIES import MIES
from cma_es import cma_es
# from surrogate import RrandomForest

from sklearn.metrics import r2_score

MACHINE_EPSILON = np.finfo(np.double).eps

class BayesOpt(object):
    """
    Generic Bayesian optimization algorithm
    """
    def __init__(self, conf_space, obj_func, eval_budget,
                 minimize=True, noisy=False, max_iter=None, 
                 wait_iter=3, n_init_sample=None, n_restart=None, 
                 verbose=False, random_seed=None):

        self._check_params()
        self.verbose = verbose
        self.init_n_eval = 1
        self.conf_space = conf_space
        self.var_names = [conf['name'] for conf in self.conf_space]
        self.obj_func = obj_func
        self.noisy = noisy

        assert hasattr(self.obj_func, '__call__')

        # random forest is used as the surrogate for now
        # TODO: add Gaussian process (OWCK) to here
        self.minimize = minimize
        self.dim = len(self.conf_space)

        self.con_ = [k['name'] for k in self.conf_space if k['type'] == 'R']  # continuous
        self.cat_ = [k['name'] for k in self.conf_space if k['type'] == 'D']  # nominal
        self.int_ = [k['name'] for k in self.conf_space if k['type'] == 'I']  # ordinal
        self.param_type = [k['type'] for k in self.conf_space]
        
        self.N_r = len(self.con_)
        self.N_d = len(self.cat_)
        self.N_i = len(self.int_)
        self.bounds = self._extract_bounds()
        
        if self.N_d == 0 and self.N_i == 0:
            self._optimizer = 'BFGS'
            lb, ub = self.bounds[0, :], self.bounds[1, :]
            thetaL = 1e-3 * (ub - lb) * np.ones(self.dim)
            thetaU = 10 * (ub - lb) * np.ones(self.dim)
            theta0 = np.random.rand(self.dim) * (thetaU - thetaL) + thetaL

            self.surrogate = GaussianProcess(regr='constant', corr='matern',
                        theta0=theta0, thetaL=thetaL,
                        thetaU=thetaU, nugget=1e-5,
                        nugget_estim=False, normalize=False,
                        verbose=False, random_start = 15 * self.dim,
                        random_state=random_seed)
        else:
            # self.surrogate = RrandomForest()
            pass
        self._optimizer = 'MIES'

        # parameter: objective evaluation
        self.max_eval = int(eval_budget)
        self.random_start = int(30 * self.dim) if n_restart is None else n_restart
        self.eval_count = 0
        self.eval_hist = []
        self.eval_hist_id = []
        self.bath_eval = False

        # optimization settings
        self.max_iter = np.inf if max_iter is None else int(max_iter)
        self.n_init_sample = self.dim * 20 if n_init_sample is None else int(n_init_sample)

        # The number of potential configuations compared against the current best
        # self.mu = int(np.ceil(self.n_init_sample / 3))
        self.mu = 3
        self.levels = [k['levels'] for k in self.conf_space if k['type'] == 'D']

        # parameter: acqusition function maximization
        self.max_eval_acquisition = 1e3 * self.dim
        self.wait_iter = wait_iter
        self.random_start_acquisition = 10

        # stop criteria
        self.stop_dict = {}
        
        # set the random seed
        self.random_seed = random_seed
        if self.random_seed:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

    def _extract_bounds(self):
        # extract variable bounds from the configuration space
        # TODO: this function should also be part of search space class
        bounds = []
        for k in self.conf_space:
            if k['type'] in ['I', 'R']:
                bounds.append(k['bounds'])
        return np.array(bounds).T

    def _better(self, perf1, perf2):
        if self.minimize:
            return perf1 < perf2
        else:
            return perf1 > perf2

    def evaluate(self, conf, runs=1):
        perf_, n_eval = conf.perf, conf.n_eval
        try:
            __ = [self.obj_func(conf[self.var_names].to_dict()) for i in range(runs)]
        except:
            __ = [self.obj_func(conf[self.var_names].as_matrix()) for i in range(runs)]
        perf = np.sum(__)

        conf.perf = perf / runs if not perf_ else np.mean((perf_ * n_eval + perf))
        conf.n_eval += runs

        self.eval_count += runs
        self.eval_hist += __
        self.eval_hist_id += [conf.index] * runs
        return conf

    def fit_and_assess(self):
        # build the surrogate model
        X, perf = self.data[self.var_names], self.data['perf']
        self.surrogate.fit(X, perf)
        
        self.is_updated = True
        perf_hat = self.surrogate.predict(X)
        r2 = r2_score(perf, perf_hat)
        if self.verbose:
            print 'Surrogate model r2: {}'.format(r2)
        return r2

    def sampling(self, N):
        # TODO: think how to do LHS for integer and categorical variable
        # use uniform random sampling for now
        data = []
        for subspace in self.conf_space:
            type_ = subspace['type']
            # TODO: move the sampling to the Search space class
            if  type_ == 'D':
                n_levels = len(subspace['levels'])
                idx = randint(0, n_levels, N)
                data.append(np.array(subspace['levels'])[idx])
            elif type_ == 'I':
                data.append(randint(subspace['bounds'][0], subspace['bounds'][1], N))
            elif type_ == 'R':
                lb, ub = subspace['bounds']
                data.append((ub - lb) * rand(N) + lb)

        data.append([0] * N)         # No. of evaluations for each solution
        data.append([None] * N)      # the corresponding objective value

        data = np.atleast_2d(data).T
        data = pd.DataFrame(data, columns=self.var_names + ['n_eval', 'perf'])

        data[self.con_] = data[self.con_].apply(pd.to_numeric)
        data[self.int_] = data[self.int_].apply(pd.to_numeric)

        return data

    def _remove_duplicate(self, confs):
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

    def select_candidate(self):
        self.is_updated = False
        # always generate mu + 1 candidate solutions
        while True:
            confs_, acqui_opts_ = self.arg_max_acquisition()
            if 1 < 2:
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
                if self._better(self.incumbent.perf, conf.perf):
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

    def optimize(self):
        if self.verbose:
            print 'building the initial design of experiemnts...'
                # create the initial data set
        self.data = self.sampling(self.n_init_sample)

        print 'evaluating the initial design sites...'
        for i, conf in self.data.iterrows():
            self.data.loc[i] = self.evaluate(conf, runs=self.init_n_eval)
            print conf.to_frame().T
        
        # set the initial incumbent
        self.data.perf = pd.to_numeric(self.data.perf)
        perf = np.array(self.data.perf)
        self.incumbent = np.nonzero(perf == np.min(perf))[0][0]
        self.fit_and_assess()

        self.iter_count = 0
        while not self.stop():
            ids = self.select_candidate()
            if self.noisy:
                self.incumbent = self.intensify(ids)
            else:
                perf = np.array(self.data.perf)
                self.incumbent = np.nonzero(perf == np.min(perf))[0][0]

            # model re-training
            self.fit_and_assess()
            self.iter_count += 1

            if self.verbose:
                print 'iteration {}, current incumbent is:'.format(self.iter_count)
                print self.data.loc[[self.incumbent]]
        
        self.stop_dict['n_eval'] = self.eval_count
        return self.incumbent

    def stop(self):
        if self.iter_count > self.max_iter:
            self.stop_dict['max_iter'] = True

        if self.eval_count > self.max_eval:
            self.stop_dict['max_eval'] = True

        return len(self.stop_dict)

    def _acquisition_func(self, plugin=None, dx=False):
        if plugin is None:
            plugin = np.min(self.data.perf) if self.minimize else np.max(self.data.perf)
        acquisition_func = EI(self.surrogate, plugin, minimize=self.minimize)
        def func(x):
            res = acquisition_func(x, dx=dx)
            return (-res[0], -res[1]) if dx else -res
        return func

    def arg_max_acquisition(self, plugin=None):
        eval_budget = self.max_eval_acquisition
        fopt = -np.inf
        optima, foptima = [], []
        wait_count = 0

        for iteration in range(self.random_start_acquisition):
            x0 = [_ for _ in self.sampling(1)[self.var_names].values[0]]
            # make sure the returned xopt_ is a list
            if self._optimizer == 'BFGS':
                obj_func = self._acquisition_func(plugin, dx=True)
                xopt_, fopt_, stop_dict = fmin_l_bfgs_b(obj_func, x0, pgtol=1e-8,
                                                        factr=1e6, bounds=self.bounds.T,
                                                        maxfun=eval_budget)
                # xopt_ = xopt_.flatten().tolist()
                if stop_dict["warnflag"] != 0 and self.verbose:
                    warnings.warn("L-BFGS-B terminated abnormally with the "
                                  " state: %s" % stop_dict)
            elif self._optimizer == 'MIES':
                obj_func = self._acquisition_func(plugin, dx=False)
                mies = MIES(obj_func, x0, self.bounds, self.levels,
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
            if eval_budget <= 0 or wait_count >= self.wait_iter:
                break
            # sort the optima in descending order
            idx = np.argsort(foptima)[::-1]
            optima = [optima[_] for _ in idx]
            foptima = [foptima[_] for _ in idx]
        return optima, foptima

    def _check_params(self):
        pass

