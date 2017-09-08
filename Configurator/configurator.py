# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:05:01 2017

@author: wangronin
"""

from __future__ import division   # important! for the float division

import pdb, warnings

import random
import numpy as np
from numpy import ceil
from numpy.random import randint, rand

import pandas as pd

from criteria import EI
from MIES import MIES

from sklearn.metrics import r2_score

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri


MACHINE_EPSILON = np.finfo(np.double).eps

# numpy and pandas data type conversion to R
numpy2ri.activate()
pandas2ri.activate()

# TODO: implement the configuration space class

# Python wrapper for the R 'randomForest' library 
class RrandomForest(object):
    def __init__(self):
        self.pkg = importr('randomForest')
        
    def fit(self, X, y):
        self.columns = X.columns
        self.n_sample, self.n_feature = X.shape
        self.rf = self.pkg.randomForest(x=X, y=y, ntree=50, 
                                        mtry=ceil(self.n_feature * 5 / 6.),
                                        nodesize=10)
        return self
    
    def predict(self, X, eval_MSE=False):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame([X], columns=self.columns)
        pdb.set_trace()
        _ = self.pkg.predict_randomForest(self.rf, X, predict_all=eval_MSE)
        if eval_MSE:
            y_hat = np.array(_[0])
            mse = np.std(np.atleast_2d(_[1]), axis=1, ddof=1) ** 2
            return y_hat, mse
        else:
            return np.array(_)
        
        
class configurator(object):
    """
    Bayesian optimization based configurator
    """
    def __init__(self, conf_space, obj_func, eval_budget, 
                 minimize=True, max_iter=None, n_init_sample=None,
                 verbose=False, random_seed=666):

        self._check_params()
        self.verbose = verbose
        self.init_n_eval = 1
        self.conf_space = conf_space
        self.var_names = [conf['name'] for conf in self.conf_space]
        self.obj_func = obj_func 
        
        assert hasattr(self.obj_func, '__call__')
        
        # random forest is used as the surrogate for now
        # TODO: add Gaussian process (OWCK) to here
        self.surrogate = RrandomForest()
        self.minimize = minimize
        self.dim = len(self.conf_space)
        
        self.con_ = [k['name'] for k in self.conf_space if k['type'] == 'R']  # continuous
        self.cat_ = [k['name'] for k in self.conf_space if k['type'] == 'D']  # nominal discrete
        self.int_ = [k['name'] for k in self.conf_space if k['type'] == 'I']  # ordinal discrete
        self.param_type = [k['type'] for k in self.conf_space]
        
        # parameter: objective evaluation
        self.max_eval = int(eval_budget)
        self.random_start = 30
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
        self.bounds = self._extract_bounds()
        self.levels = [k['levels'] for k in self.conf_space if k['type'] == 'D']

        # stop criteria
        self.stop_condition = []

        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
    def _extract_bounds(self):
        # extract variable bounds from the configuration space
        bounds = []
        for k in self.conf_space:
            if k['type'] in ['I', 'R']:
                bounds.append(k['bounds'])
        return bounds
        
    def _better(self, perf1, perf2):
        if self.minimize:
            return perf1 < perf2
        else:
            return perf1 > perf2
        
    def evaluate(self, conf, runs=1):
        perf_, n_eval = conf.perf, conf.n_eval
        __ = [self.obj_func(conf[self.var_names].to_dict()) for i in range(runs)]
        perf = np.sum(__)
        
        conf.perf = perf / runs if not perf_ else np.mean((perf_ * n_eval + perf))
        conf.n_eval += runs
        
        self.eval_count += runs
        self.eval_hist += __
        self.eval_hist_id += [conf.name] * runs
        return conf
        
    def fit_and_assess(self):
        # build the surrogate model
        X, perf = self.data[self.var_names], self.data['perf']
        self.surrogate.fit(X, perf)
        
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
            CON = np.all(np.isclose(X[self.con_], x[self.con_], axis=1))
            INT = np.all(X[self.int_] == x[self.int_], axis=1)
            CAT = np.all(X[self.cat_] == x[self.cat_], axis=1)
            if not (CON and INT and CAT):
                idx.append(i)
        return confs.iloc[idx, :]
        
    def select_candidate(self):
        # always generate mu + 1 candidate solutions
        while True:
            confs_, acqui_opts_ = self.arg_max_acquisition()
            N = self.data.shape[0]
            confs_ = pd.DataFrame([[N + i, None, 0] + conf for i, conf in enumerate(confs_)], 
                                   columns=['id', 'perf', 'n_eval'] + self.var_names)
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
                    confs_ = self.sampling(n=1)
                    break
            else:
                break
        
        # proportional selection without replacement
        candidates_id = list(confs_.id)
        if 1 < 2:
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

        self.data.append(confs_)
        for i, conf in self.confs_.iterrows():
            self.evaluate(conf, runs=self.init_n_eval)

        return candidates_id

    def intensify(self, candidates_ids):
        maxR = 20 # maximal number of the evaluations on the incumbent
        for i, ID in enumerate(candidates_ids):
            r, extra_run = 1, 1
            # ID is not the dataframe index...
            conf = self.data[self.data.id == ID]
            self.evaluate(conf, 1)
            
            if conf.n_eval > self.incumbent.n_eval:
                self.evaluate(self.incumbent, 1)
                extra_run = 0

            while True:
                if self._better(self.incumbent.perf, conf.perf):
                    self.evaluate(self.incumbent, min(extra_run, maxR - self.incumbent.n_eval))
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
                self.evaluate(conf, r)
                extra_run += r
        
    def configure(self):
        if self.verbose:
            print 'building the initial design of experiemnts...'
        # create the initial data set
        self.data = self.sampling(self.n_init_sample)
        
        print 'evaluating the initial design sites...'
        for i, conf in self.data.iterrows():
            self.data.loc[i] = self.evaluate(conf, runs=self.init_n_eval)
            print conf.to_frame().T
        
        self.data.perf = pd.to_numeric(self.data.perf)
            
        # set the initial incumbent
        perf = np.array(self.data.perf)
        idx = np.nonzero(perf == np.max(perf))[0][0]
        self.incumbent = self.data.iloc[idx, :]
        self.fit_and_assess()
        
        self.iter_count = 0
        while not self.stop():
            ids = self.select_candidate()
            self.incumbent = self.intensify(ids)

            self.fit_and_assess()
            self.iter_count += 1

        return self.incumbent
    
    def stop(self):
        if self.iter_count > self.max_iter:
            self.stop_condition.append('max_iter')

        if self.eval_count > self.max_eval:
            self.stop_condition.append('max_eval')

        return len(self.stop_condition)
        
    def arg_max_acquisition(self, plugin=None):
        if plugin is None:
            plugin = np.min(self.data.perf) if self.minimize else np.max(self.data.perf)

        obj_func = EI(self.surrogate, plugin, minimize=self.minimize)
        
        eval_budget = 1e4 * self.dim
        xopt, fopt = None, -np.inf
        for iteration in range(self.random_start):
            
            x0 = [_ for _ in self.sampling(1)[self.var_names].values[0]]
            pdb.set_trace()
            mies = MIES(obj_func, x0, self.bounds, self.levels, 
                        self.param_type, eval_budget, minimize=False)
            xopt_, fopt_, stop_dict = mies.optimize()
            
            # TODO: verify this rule to determine the insignificant improvement 
            diff = (fopt_ - fopt) / max(abs(fopt_), abs(fopt), 1)  
            if diff >= 1e7 * MACHINE_EPSILON:
                xopt, fopt = xopt_, fopt_
                wait_count = 0
            else:
                wait_count += 1

            if self.verbose:
                print 'restart {} takes {} evals'.format(iteration + 1, stop_dict['n_evals'])
                print 'best acquisition function values: {}'.format(fopt)

            eval_budget -= stop_dict['n_evals']
            if eval_budget <= 0 or wait_count >= self.wait_iter:
                break

        return xopt, fopt
    
    def _check_params(self):
        pass


#    def __init_obj_func(self):
#        if self.verbose:
#            print 'Creating the objective function for the configuration...'
#
#        def __(par):
#            par_, perf_ = par['par'], par['perf']
#            # evaluate the par(ameters) by fitting the model
#            model_par = {key : par_[i] for i, key in enumerate(self.par_list)}
#            model = regressor(algorithm=self.algorithm, metric=self.metric,
#                              model_par=model_par, light_mode=True,
#                              verbose=False, random_seed=self.random_seed)
#
#            model.fit(self.X, self.y, n_fold=self.n_fold, parallel=False)
#
#            # TODO: cross-validation may also be considered as multiple problem instance
#            perf = np.mean(model.performance[self.metric])
#
#            N = par['N']
#            par['perf'] =  perf if perf_ is None else (perf_ * N + perf) / (N + 1)
#            par['N'] += 1
#
#            self.__current_model = model
#            self.eval_count += 1
#            return perf
#        return __
