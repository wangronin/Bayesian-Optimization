# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:05:01 2017

@author: wangronin
"""

from __future__ import division   # important! for the float division

import pdb, sys

import numpy as np
from numpy import sin, cos, pi, inf

import random
import pandas as pd
import redis

from regression import regressor
from ego import ego

from GaussianProcess import GaussianProcess_extra as GaussianProcess

class configurator:

    def __init__(self, n_iter, algorithm, metric='mse', n_fold=10,
                 par_list=None, par_lb=None, par_ub=None, verbose=False,
                 conn=None, save_tmp=False, ID='', random_seed=999):

        self.level = 0
        self.algorithm = algorithm
        self.verbose = verbose
        self.metric = metric
        self.n_fold = n_fold
        self.conn = conn
        self.ID = ID

        # parameter to configure
        self.par_list = par_list
        self.par_lb = np.atleast_1d(par_lb)
        self.par_ub = np.atleast_1d(par_ub)
        self.dim = len(par_list)

        # optimization settings
        # TODO: these parameters should be properly set
        self.n_iter = n_iter
        self.n_init_sample = 10
        self.obj_func = self.__obj_func()
        self.stop_condition = []

        if self.conn is not None:
            self.__db_conn = redis.Redis(**self.conn)

        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def check_Xy(self, X, y):
        # Checker for the dimensionality of the data
        # TODO: implement this
        return X, y

    def __obj_func(self):

        def __(par):
            model_par = {key : par[i] for i, key in enumerate(self.par_list)}
            model = regressor(algorithm=self.algorithm, metric=self.metric,
                              model_par=model_par, random_seed=1,
                              verbose=False)

            model.fit(self.X, self.y, n_fold=self.n_fold, parallel=False)

            # median is more reliable for CV
            score = np.mean(model.performance[self.metric])
            self.__current_model = model

            return score
        return __

    def get_r2(self):
        from sklearn.metrics import r2_score
        X = self.conf_optimizer.X
        y = self.conf_optimizer.y

        y_hat = self.gp.predict(X)
        r2 = r2_score(y, y_hat)

        return r2

#    def plot_convergence(self, ax=None):
#        t += .05
#        y = -2*sin(x)*sin(t)
#        line.set_ydata(y)
#
#        lines[0].set_data(x2, y2)
#        lines[1].set_data(x2, y2*.01)
#
#        # In matplotlib 1.4, you have to do this for dynamic plotting
#        # calling draw does not  produce the plot
#        f.canvas.draw()
#        f.canvas.flush_events()

    def configure(self, X, y):
        import matplotlib as mpl
        mpl.use('TKAgg')

        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        from pandas.plotting import parallel_coordinates
        plt.ion()

        rcParams['legend.numpoints'] = 1
        rcParams['xtick.labelsize'] = 15
        rcParams['ytick.labelsize'] = 15
        rcParams['xtick.major.size'] = 7
        rcParams['xtick.major.width'] = 1
        rcParams['ytick.major.size'] = 7
        rcParams['ytick.major.width'] = 1
        rcParams['axes.labelsize'] = 15

        fig_width = 22
        fig_height = 22 * 9 / 16
        f, (ax0, ax1) = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=100)
        f.suptitle('Model Configurator')

        ax0.grid(True)
        ax1.grid(True)
        ax0.set_title('optimization convergence')
        ax1.set_title('model parameter')

        ax0.set_ylabel(r'Model quality MSE')
        ax0.set_xlabel(r'iteration')

        line0, = ax0.plot([], [], ls='--', marker='^', color='r',
                          ms=6, mfc='none', mec='r', alpha=0.8)
        line1, = ax0.plot([], [], ls='-', marker='o', color='k',
                          ms=6, mfc='none', mec='k', alpha=0.8)
        ax0.legend(['new solution quality', 'current best'])

        X, y = self.check_Xy(X, y)
        self.X, self.y = X, y

        thetaL = 1e-3 * (self.par_ub - self.par_lb) * np.ones(self.dim)
        thetaU = 10 * (self.par_ub - self.par_lb) * np.ones(self.dim)
        theta0 = np.random.rand(self.dim) * (thetaU - thetaL) + thetaL

        if self.verbose:
            print 'building the initial design of experiemnts...'

        # meta-model: Gaussian process regression
        self.__conf_gp = GaussianProcess(regr='constant', corr='matern',
                                         theta0=theta0, thetaL=thetaL,
                                         thetaU=thetaU, nugget=1e-5,
                                         nugget_estim=False, normalize=False,
                                         verbose=False, random_start = 15*self.dim,
                                         random_state=self.random_seed)

        # efficient global optimization algorithm
        self.__conf_optimizer = ego(self.dim, self.obj_func,
                                    self.__conf_gp,
                                    self.n_iter,
                                    lb=self.par_lb, ub=self.par_ub,
                                    doe_size=self.n_init_sample,
                                    n_restart=50*self.dim,
                                    solver='BFGS', verbose=False)

        self.conf_optimizer = self.__conf_optimizer
        self.gp = self.conf_optimizer.model

        r2 = self.get_r2()
        print 'initial GPR model R2: ', r2

        # TODO: implement more stop criteria
        self.par_opt = None
        self.metric_value_opt = inf

        for n in range(self.n_iter):
            _, __, par_, metric_value_, ei_value = self.__conf_optimizer.step()
            par_ = par_.flatten()

            if metric_value_ < self.metric_value_opt:
                	self.par_opt, self.metric_value_opt = par_.flatten(), metric_value_
                	self.model_opt = self.__current_model

            if n == 0:
                par_list = [list(np.log(par_)) + ['new solution']]
                par_list.append(list(np.log(self.par_opt)) + ['current optimal'])
            else:
                del par_list[-1]
                par_list.append(list(np.log(par_)) + ['new solution'])
                par_list.append(list(np.log(self.par_opt)) + ['current optimal'])

            df = pd.DataFrame(par_list, columns=self.par_list + ['Name'])
            x, y = line0.get_data()
            x = np.r_[x, n+1]
            y = np.r_[y, np.log(metric_value_)]
            line0.set_data(x, y)

            if len(y) > 15:
                max_y = min(max(y[-15:]), 5)
            else:
                max_y = max(y)

            x, y = line1.get_data()
            x = np.r_[x, n+1]
            y = np.r_[y, np.log(self.metric_value_opt)]
            line1.set_data(x, y)

            if len(y) > 15:
                min_y = min(y[-15:])
            else:
                min_y = min(y)

            ax0.set_xlim(min(x), max(x))
            ax0.set_ylim(min_y * .99, max_y * 1.01)

            parallel_coordinates(df, 'Name', ax=ax1)
            ax1.legend_.remove()
            ax1.legend(['new solution quality', 'current best'])

            f.canvas.draw()
            f.canvas.flush_events()

            if self.verbose:
                print '+' + '-' * 80 + '+'
                print '[DEBUG] iteration {}:'.format(n+1)
                print '[DEBUG] new candidate param: ', par_
                print '[DEBUG] with EI: {}, {}: {} '.format(ei_value,
                       self.metric, metric_value_)
                print
                print '[DEBUG] optimal model performance {}: {}'.format(self.metric,
                       self.metric_value_opt)
                print '[DEBUG] optimal param: {}'.format(self.par_opt)
                print '[DEBUG] After re-estimation GPR R2: {}'.format(self.get_r2())
                print

            if hasattr(self, '_configurator' + '__db_conn'):
                self.__db_conn.lpush(self.ID + 'metric_value_opt', self.metric_value_opt)
                for i in range(self.dim):
                    self.__db_conn.lpush(self.ID + 'par_opt{}'.format(n+1), self.par_opt[i])

        else:
            self.stop_condition.append('max_iter')

        if hasattr(self, '_configurator' + '__db_conn'):
            self.__db_conn.lpush(self.ID + 'metric_value_opt_final',
                                 self.metric_value_opt[0])
            for i in range(self.dim):
                self.__db_conn.lpush(self.ID + 'par_opt_final{}'.format(n+1),
                                     self.par_opt[i])

        return self.par_opt, self.metric_value_opt, self.model_opt

if __name__ == '__main__':

    np.random.seed(123)
    from fitness import rastrigin

    # test problem: to fit a so-called Rastrigin function in 20D
    X = np.random.rand(1000, 20)
    y = rastrigin(X.T)

#    y = (y - y.mean()) / y.std()

    # to configure a SVM regression model
    conf = configurator(100, 'SVM', 'mse', 10,
                        ['C', 'epsilon', 'gamma'],
                        [1e-20, 1e-10, 1e-10],
                        [30, 1, 5], verbose=True,
                        random_seed=123)
#                        conn={'host' : 'localhost',
#                              'port' : 6379,
#                              'password' : ''})

    res = conf.configure(X, y)