#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:10:18 2017

@author: wangronin
"""

import pdb

from pandas import DataFrame

import numpy as np
from numpy import exp, nonzero, argsort
from numpy.random import randint, rand, randn, geometric

from constraint import boundary_handling

class MIES(object):

    def __init__(self, obj_func, x0, bounds, levels, param_type, max_eval,
                 minimize=True, mu_=4, lambda_=28, sigma0=1, eta0=0.05, P0=0.2,
                 verbose=False):

        self.verbose = verbose
        self.mu_ = mu_
        self.lambda_ = lambda_
        self.eval_count = 0
        self.iter_count = 0
        self.max_eval = max_eval
        self.param_type = param_type
        self.plus_selection = False
        self.levels = levels

        # index of each type of variables in the dataframe
        self.id_r = nonzero(np.array(self.param_type) == 'R')[0]
        self.id_i = nonzero(np.array(self.param_type) == 'I')[0]
        self.id_d = nonzero(np.array(self.param_type) == 'D')[0]

        # the number of variables per each type
        self.N_r = len(self.id_r)
        self.N_i = len(self.id_i)
        self.N_d = len(self.id_d)
        self.dim = self.N_r + self.N_i + self.N_d

        # by default, we use individual step sizes for continuous and integer variables
        # and global strength for the nominal variables
        N_p = min(self.N_d, int(1))

        bounds = np.atleast_2d(bounds)
        self.bounds = bounds.T if bounds.shape[0] != 2 else bounds

        self.minimize = minimize
        self.obj_func = obj_func
        self.stop_dict = {}

        # initialize the populations
        fitness0 = self.obj_func(x0)
        individual0 = x0 + [sigma0] * self.N_r + [eta0] * self.N_i + [P0] * N_p + [fitness0]
        self.xopt = x0
        self.fopt = fitness0

        # column names of the dataframe: used for slicing
        self.cols_x = ['x{}'.format(i) for i in range(self.dim)]
        self.cols_sigma = ['sigma{}'.format(i) for i in range(self.N_r)]
        self.cols_eta = ['eta{}'.format(i) for i in range(self.N_i)]
        self.cols_p = ['P{}'.format(i) for i in range(N_p)]
        self.cols = self.cols_x + self.cols_sigma + self.cols_eta + self.cols_p + ['fitness']

        self.pop_mu = DataFrame([individual0] * self.mu_, columns=self.cols)
        self.pop_lambda = DataFrame([individual0] * self.lambda_, columns=self.cols)
        self.par_id = range(self.dim)
        self.s_par_id = range(self.dim, len(self.cols) - 1)

        self._set_hyperparam()

    def _set_hyperparam(self):
        # hyperparameters: mutation strength adaptation
        self.tau_r = 1 / np.sqrt(2 * self.N_r)
        self.tau_p_r = 1 / np.sqrt(2 * np.sqrt(self.N_r))

        self.tau_i = 1 / np.sqrt(2 * self.N_i)
        self.tau_p_i = 1 / np.sqrt(2 * np.sqrt(self.N_i))

        self.tau_d = 1 / np.sqrt(2 * self.N_d)
        self.tau_p_d = 1 / np.sqrt(2 * np.sqrt(self.N_d))

    def keep_in_bound(self, pop):
        idx = np.r_[self.id_r, self.id_i]
        X = pop.iloc[:, idx]
        pop.iloc[:, idx] = boundary_handling(X, self.bounds[0, idx], self.bounds[1, idx])
        pop.iloc[:, self.id_i] = pop.iloc[:, self.id_i].applymap(int)
        return pop

    def recombine(self, id1, id2):
        p1 = self.pop_mu.loc[id1].copy()
        p1.fitness = None
        if id1 != id2:
            p2 = self.pop_mu.loc[id2].copy()
            p1[self.s_par_id] = (p1[self.s_par_id] + p2[self.s_par_id]) / 2
            for index in self.par_id:
                if rand() > 0.5:
                    p1[index] = p2[index]
        return p1

    def select(self):
        pop = self.pop_mu.append(self.pop_lambda) if self.plus_selection else self.pop_lambda

        fitness_rank = argsort(pop.fitness.as_matrix())
        if not self.minimize:
            fitness_rank = fitness_rank[::-1]

        sel_id = fitness_rank[:self.mu_]
        self.pop_mu = pop.iloc[sel_id, :].copy()
        self.pop_mu.index = range(self.mu_)

    def evaluate(self, pop):
        for i, individual in pop.iterrows():
            par = individual[self.par_id]
            pop.loc[i, 'fitness'] = self.obj_func(par)
            self.eval_count += 1

    def mutate(self, individual):
        if self.N_r:
            self._mutate_r(individual)
        if self.N_i:
            self._mutate_i(individual)
        if self.N_d:
            self._mutate_d(individual)
        return individual

    def _mutate_r(self, individual):
        sigma = np.array(list(individual[self.cols_sigma].values))
        if len(self.cols_sigma) == 1:
            sigma = sigma * exp(self.tau_r * randn())
        else:
            sigma = sigma * exp(self.tau_p_r * randn() + self.tau_r * randn(self.N_r))

        individual[self.cols_sigma] = sigma
        individual[self.id_r] += sigma * randn(self.N_r)

    def _mutate_i(self, individual):
        eta = np.array(list(individual[self.cols_eta].values))
        if len(self.cols_eta) == 1:
            eta = max(1, eta * exp(self.tau_i * randn()))
            p = 1 - (eta / self.N_i) / (1 + np.sqrt(1 + (eta / self.N_i) ** 2))
            individual[self.id_i] += geometric(p, self.N_i) - geometric(p, self.N_i)
        else:
            eta = max(1, eta * exp(self.tau_p_i * randn() + self.tau_i * randn(self.N_i)))
            p = 1 - (eta / self.N_i) / (1 + np.sqrt(1 + (eta / self.N_i) ** 2))
            individual[self.id_i] += [geometric(p_) - geometric(p_) for p_ in p]

        individual[self.cols_eta] = eta

    def _mutate_d(self, individual):
        P = np.array(list(individual[self.cols_p].values))
        P = 1 / (1 + (1 - P) / P * exp(-self.tau_d * randn()))
        individual[self.cols_p] = boundary_handling(P, 1 / (3. * self.N_d), 0.5)

        idx = np.nonzero(rand(self.N_d) < P)[0]
        for i in range(idx):
            level = self.levels[i]
            individual[self.id_d[i]] = level[randint(0, len(level))]
        # if sum(idx):
        #     bounds_d = self.bounds[:, self.id_d][:, idx].T
        #     levels = [level for i, level in enumerate(self.levels) if idx[i]]
        #     individual[self.id_d[idx]] = [level[randint(0, len(level))] for i, level in levels]
        #     individual[self.id_d[idx]] = [randint(b[0], b[1]) for b in bounds_d]

    def stop(self):
        if self.eval_count > self.max_eval:
            self.stop_dict['max_eval'] = True
        return len(self.stop_dict)

    def _better(self, perf1, perf2):
        if self.minimize:
            return perf1 < perf2
        else:
            return perf1 > perf2

    def optimize(self):
        while not self.stop():
            for i in range(self.lambda_):
                p1, p2 = randint(0, self.mu_), randint(0, self.mu_)

                individual = self.recombine(p1, p2)
                self.pop_lambda.loc[i] = self.mutate(individual)

            self.keep_in_bound(self.pop_lambda)
            self.evaluate(self.pop_lambda)
            self.select()

            curr_best = self.pop_mu.loc[0]
            xopt_, fopt_ = curr_best[self.par_id].as_matrix(), curr_best.fitness
            self.iter_count += 1

            if self._better(fopt_, self.fopt):
                self.xopt, self.fopt = xopt_, fopt_

            if self.verbose:
                print self.xopt, self.fopt

        self.stop_dict['n_evals'] = self.eval_count
        return self.xopt, self.fopt


if __name__ == '__main__':

    np.random.seed(1)
    def fitness(x):
        x_r, x_i, x_d = np.array(x[:2]), x[2], x[3]
        if x_d == 'OK':
            tmp = 0
        else:
            tmp = 1
        return np.sum(x_r ** 2) + abs(x_i - 10) / 123. + tmp * 2

    x0 = [2, 1, 80, 'B']
    bounds = [[-5] * 2 + [-100], [5] * 2 + [100]]
    levels = [['OK', 'A', 'B', 'C', 'D', 'E', 'F', 'G']]

    opt = MIES(fitness, x0, bounds, levels, ['R', 'R', 'I', 'D'], 1e3, verbose=True)
    print opt.optimize()
