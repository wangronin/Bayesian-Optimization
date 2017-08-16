# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:05:01 2017

@author: wangronin
"""

from __future__ import division   # important! for the float division

import pdb, warnings
from copy import copy

import numpy as np

from regression import regressor
from ego import ego

from owck import GaussianProcess_extra as GaussianProcess

import random
from pyDOE import lhs

class metric:

    def __init__(self):
        pass

    def __call__(self):
        pass

class configurator:

    def __init__(self, algorithm, metric, budget,
                 data=None, max_iter=None, n_init_sample=None,
                 par_list=None, par_lb=None, par_ub=None, n_fold=10,
                 verbose=False, random_seed=999):

        self.algorithm = algorithm
        self.verbose = verbose
        self.metric = metric
        self.n_fold = n_fold
        self.model_type = 'GPR'
        self.init_N = 1

        # parameter: evaluation
        self.max_eval = int(budget)
        self.eval_count = 0
        self.eval_hist = []
        self.eval_hist_id = []
        self.bath_eval = False

        # TODO: to complete this
        self._metrics = ['mse', 'r2', 'auc']
        self.is_minimize = True if self.metric in ['mse'] else False

        # parameter to configure
        self.par_list = par_list
        self.par_lb = np.atleast_1d(par_lb)
        self.par_ub = np.atleast_1d(par_ub)
        self.dim = len(par_list)

        # optimization settings
        # TODO: these parameters should be properly set
        self.max_iter = np.inf if max_iter is None else int(max_iter)
        self.DOE_size = self.dim * 20 if n_init_sample is None else int(n_init_sample)

        # The number of potential configuations compared against the current best
#        self.mu = int(np.ceil(self.DOE_size / 3))
        self.mu = 3

        if data is not None:
            self.set_data(data[0], data[1])

        # stop criteria
        self.stop_condition = []

        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def check_Xy(self, X, y):
        # TODO: why do I want to implement here?
        return X, y

    def set_data(self, X, y):
        X, y = self.check_Xy(X, y)
        self.X, self.y = X, y

        # immediately update the objective function after updating the data
        self.obj_func = self.__init_obj_func()

    def __init_obj_func(self):

        if self.verbose:
            print 'Creating the objective function for the configuration...'

        def __(par):
            par_, perf_ = par['par'], par['perf']
            # evaluate the par(ameters) by fitting the model
            model_par = {key : par_[i] for i, key in enumerate(self.par_list)}
            model = regressor(algorithm=self.algorithm, metric=self.metric,
                              model_par=model_par, light_mode=True,
                              verbose=False, random_seed=self.random_seed)

            model.fit(self.X, self.y, n_fold=self.n_fold, parallel=False)

            # TODO: cross-validation may also be considered as multiple problem instance
            perf = np.mean(model.performance[self.metric])

            N = par['N']
            par['perf'] =  perf if perf_ is None else (perf_ * N + perf) / (N + 1)
            par['N'] += 1

            self.__current_model = model
            self.eval_count += 1
            return perf

        return __

    def evaluate(self, par, runs):
        self.eval_hist += [self.obj_func(par) for i in range(runs)]
        self.eval_hist_id += [par['id']] * runs
        self.eval_count += runs

    def __preparation(self):

        if self.verbose:
            print 'building the initial design of experiemnts...'

        # TODO: double check the hyperparameter settings for GPR
        thetaL = 1e-1 * (self.par_ub - self.par_lb) * np.ones(self.dim)
        thetaU = 10 * (self.par_ub - self.par_lb) * np.ones(self.dim)
        theta0 = np.random.rand(self.dim) * (thetaU - thetaL) + thetaL

        if self.model_type == 'GPR':
            # meta-model: Gaussian process regression
            self.gpr = GaussianProcess(regr='constant', corr='matern',
                                         theta0=theta0, thetaL=thetaL,
                                         thetaU=thetaU, nugget=1e-5,
                                         nugget_estim=False, normalize=True,
                                         verbose=False, random_start = 100*self.dim,
                                         random_state=self.random_seed)
        elif self.model_type == 'RF':
            pass

        # generate initial design
        self.DOE = lhs(self.dim, samples=self.DOE_size,
                       criterion='cm') * (self.par_ub - self.par_lb) + self.par_lb

        # configurations
        self.pars = [{'id': i,
                      'par': design,
                      'perf': None,
                      'N': 0} for i, design in enumerate(self.DOE)]

        for par in self.pars:
            self.evaluate(par, self.init_N)

        perf = self.get_perf()
        idx = np.nonzero(perf == np.min(perf))[0][0]
        self.incumbent = self.pars[idx]

        # TODO: check the performance of the initial GP model: applying transformation
        # when the performance is poor
        self.gpr.fit(self.DOE, perf)
        perf_hat = self.gpr.predict(self.DOE)
        from sklearn.metrics import r2_score
        r2 = r2_score(perf, perf_hat)

        if self.verbose:
            print 'inital {} model r2: {}'.format(self.model_type, r2)

        # efficient global optimization algorithm
        self.__ego_opt = ego(self.dim, self.obj_func,
                             self.gpr, self.max_iter,
                             lb=self.par_lb, ub=self.par_ub,
                             doe_size=self.DOE_size,
                             solver='BFGS', verbose=False)

    def get_perf(self):
        return np.array([par['perf'] for par in self.pars])

    def get_pars(self):
        return np.array([par['par'] for par in self.pars])

    def __remove_duplicate(self, par_):

        par_ = np.atleast_2d(par_)
        samples = []
        X = np.array([par['par'] for par in self.pars])

        for x in par_:
            if not any(np.sum(np.isclose(X, x), axis=1)):
                samples.append(x)

        return np.array(samples)

    def select_candidate(self):
        # always generate mu + 1 candidate solutions
        while True:
            new_par, ei_value, _, __ = self.__ego_opt.max_criterion()

            # check for potential duplications
            new_par = self.__remove_duplicate(new_par).flatten()

            # if no new design site found, re-estimate the parameters immediately
            if len(new_par) == 0:
                if not self.__ego_opt.is_updated:
                    # Duplication are commonly encountered in the 'corner'
                    self.__ego_opt.update_model(self.X, self.y, re_estimation=True)
                else:
                    # getting duplcations by this is of 0 measure...
                    warnings.warn('iteration {}: duplicated solution found \
                                  by optimization! New points is taken from random \
                                  design'.format(self.iter_count))
                    new_par = np.random.rand(self.dim) * \
                        (self.par_ub - self.par_lb) + self.par_lb
                    break
            else:
                break
        print
        print 'best ', self.incumbent['par']
        print 'new solution ', new_par, ei_value
        print

        # unevaluated new canditates
        new_par = {'id': len(self.pars),
                   'par': new_par,
                   'perf': None,
                   'N': 0}

        # proportional selection without replacement
        candidates = [new_par]
        if 11 < 2:
            metrics = -np.array([par['perf'] for par in self.pars \
                                 if par['id'] != self.incumbent['id']])
            metrics -= np.min(metrics)
            index_curr = np.array([par['id'] for par in self.pars \
                                 if par['id'] != self.incumbent['id']])

            idx = np.argsort(metrics)
            metrics = metrics[idx]
            index_curr = index_curr[idx]

            for i in range(self.mu):
                min_ = np.min(metrics)
                prob = np.cumsum((metrics - min_) / (np.sum(metrics) - min_ * len(metrics)))
                _ = np.nonzero(np.random.rand() <= prob)[0][0]

                candidates.append(self.pars[index_curr[_]])
                metrics = np.delete(metrics, _)
                index_curr = np.delete(index_curr, _)

        self.pars.append(new_par)
        self.evaluate(new_par, self.init_N)

        return candidates

    def intensify(self, incumbent, candidates):

        maxR = 20 # prevent the incumbent being overly evaluated

        for i, par in enumerate(candidates):
            r, extra_run = 1, 1
            self.evaluate(par, 1)
            if par['N'] > self.incumbent['N']:
                self.evaluate(self.incumbent, 1)
                extra_run = 0

            while True:
                if par['perf'] > self.incumbent['perf']:
                    self.evaluate(self.incumbent,
                                  min(extra_run, maxR - self.incumbent['N']))
                    break
                if par['N'] > self.incumbent['N']:
                    self.incumbent = self.par
                    break

                r = min(2 * r, self.incumbent['N'] - par['N'])
                self.evaluate(par, r)
                extra_run += r

#    def update_model(self):
#        X = np.array([par['par'] for par in self.pars])
#        perf = np.array([par['perf'] for par in self.pars]).reshape(-1, 1)
#       re_estimate = self.check_re_estimation(new_par['par'], ei_value)

#        print self.__ego_opt.model.X.shape

    def stop_check(self):
        if self.iter_count > self.max_iter:
            self.stop_condition.append('max_iter')

        if self.eval_count > self.max_eval:
            self.stop_condition.append('max_eval')

        # TODO: more termination criteria

        if len(self.stop_condition) == 0:
            return True
        else:
            return False

    def configure(self):

        self.__preparation()

        self.par_opt = None
        self.metric_value_opt = np.inf

        self.iter_count = 0
        while self.stop_check():

            candidates = self.select_candidate()
#            self.intensify(self.incumbent, candidates)

            par_ = candidates[0]['par']
            metric_value_ = candidates[0]['perf']
            if metric_value_ < self.metric_value_opt:
                	self.par_opt, self.metric_value_opt = par_.flatten(), metric_value_
                	self.model_opt = self.__current_model

            self.__ego_opt.X = self.get_pars()
            self.__ego_opt.y = self.get_perf()
            self.__ego_opt.update_model(re_estimation=True)

            self.iter_count += 1

            if self.verbose:
                print '[DEBUG] iteration {} -- optimal model with \
                    {}: {}'.format(self.iter_count, self.metric, self.incumbent['perf'])
                print '[DEBUG] optimal param: {}'.format(self.incumbent['par'])
#                print [par['N'] for par in self.pars]
#                print self.incumbent['N']

        return self.incumbent, self.__current_model

if __name__ == '__main__':

    np.random.seed(1)
    from fitness import rastrigin

    # test problem: to fit a so-called Rastrigin function in 20D
    X = np.random.rand(500, 20)
    y = rastrigin(X.T)

    # configure a SVM regression model
    conf = configurator('SVM', 'mse', 1e3, data=(X, y),
                        n_init_sample=10,
                        par_list=['C', 'epsilon', 'gamma'],
                        par_lb=[1e-20, 1e-20, 1e-5],
                        par_ub=[20, 1, 2],
                        n_fold=10, verbose=True,
                        random_seed=1)

    res = conf.configure()

    pdb.set_trace()

