#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:10:18 2017

@author: wangronin
"""

import pdb

import numpy as np
from numpy import exp, nonzero, argsort, ceil, zeros, mod
from numpy.random import randint, rand, randn, geometric

from ..utils import boundary_handling

class Individual(list):
    """Make it possible to index Python list object using the enumerables
    """
    def __getitem__(self, keys):
        if isinstance(keys, int):
            return super(Individual, self).__getitem__(keys)
        elif hasattr(keys, '__iter__'):
            return Individual([super(Individual, self).__getitem__(int(key)) for key in keys])
    
    def __setitem__(self, index, values):
        if isinstance(index, int):
            if hasattr(values, '__iter__'):
                if len(values) == 1:
                    values = values[0]
                else:
                    values = Individual([_ for _ in values])
            super(Individual, self).__setitem__(index, values)
        
        elif len(index) == 1:
            index = index[0]
            if hasattr(values, '__iter__'):
                if len(values) == 1:
                    values = values[0]
                else:
                    values = Individual([_ for _ in values])
            super(Individual, self).__setitem__(index, values)
        else:
            if not hasattr(values, '__iter__'):
                values = [values]
            try:
                for i, k in enumerate(index):
                    super(Individual, self).__setitem__(k, values[i])
            except:
                pdb.set_trace()

    def __add__(self, other):
        return Individual(list.__add__(self, other))

    def __mul__(self, other):
        return Individual(list.__mul__(self, other))

    def __rmul__(self, other):
        return Individual(list.__mul__(self, other))

# TODO: improve efficiency, e.g. compile it with cython
class mies(object):
    def __init__(self, x0, obj_func, bounds, levels, param_type, max_eval,
                 minimize=True, mu_=4, lambda_=28, sigma0=0.1, eta0=0.05, P0=0.4,
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
        self.id_r = nonzero(np.array(self.param_type) == 'C')[0]
        self.id_i = nonzero(np.array(self.param_type) == 'O')[0]
        self.id_d = nonzero(np.array(self.param_type) == 'N')[0]

        # the number of variables per each type
        self.N_r = len(self.id_r)
        self.N_i = len(self.id_i)
        self.N_d = len(self.id_d)
        self.dim = self.N_r + self.N_i + self.N_d

        # by default, we use individual step sizes for continuous and integer variables
        # and global strength for the nominal variables
        N_p = min(self.N_d, int(1))
        self.bounds = self._check_bounds(bounds)

        self.minimize = minimize
        self.obj_func = obj_func
        self.stop_dict = {}

        # initialize the populations
        fitness0 = np.sum(self.obj_func(x0)) # to get a scalar value
        individual0 = Individual(x0 + [sigma0] * self.N_r + [eta0] * self.N_i + [P0] * N_p)
        self.xopt = x0
        self.fopt = fitness0

        # column names of the dataframe: used for slicing
        self._id_var = np.arange(self.dim)
        self._id_sigma = np.arange(self.N_r) + self._id_var[-1] + 1 if self.N_r else []
        self._id_eta = np.arange(self.N_i) + self._id_sigma[-1] + 1 if self.N_i else []
        self._id_p = np.arange(N_p) + self._id_eta[-1] + 1 if N_p else []
        self._id_hyperpar = np.arange(self.dim, len(individual0))
        # self.par_id = range(self.dim)
        
        # self.cols = self.cols_x + self.cols_sigma + self.cols_eta + self.cols_p

        # initialize the population
        self.pop_mu = Individual([individual0]) * self.mu_
        self.pop_lambda = Individual([individual0]) * self.lambda_
        self.f_mu = np.repeat(fitness0, self.mu_)

        self._set_hyperparameter()

        # stop criteria
        self.tolfun = 1e-5
        self.nbin = int(3 + ceil(30. * self.dim / self.lambda_))
        self.histfunval = zeros(self.nbin)
        
    def _check_bounds(self, bounds):
        bounds = np.atleast_2d(bounds)
        bounds = bounds.T if bounds.shape[0] != 2 else bounds
        if any(bounds[0, :] >= bounds[1, :]):
            raise ValueError('lower bounds are bigger than the upper bounds')
        return bounds
        
    def _set_hyperparameter(self):
        # hyperparameters: mutation strength adaptation
        if self.N_r:
            self.tau_r = 1 / np.sqrt(2 * self.N_r)
            self.tau_p_r = 1 / np.sqrt(2 * np.sqrt(self.N_r))

        if self.N_i:
            self.tau_i = 1 / np.sqrt(2 * self.N_i)
            self.tau_p_i = 1 / np.sqrt(2 * np.sqrt(self.N_i))

        if self.N_d:
            self.tau_d = 1 / np.sqrt(2 * self.N_d)
            self.tau_p_d = 1 / np.sqrt(2 * np.sqrt(self.N_d))

    def keep_in_bound(self, pop):
        idx = np.sort(np.r_[self.id_r, self.id_i])
        X = np.array([var[idx] for var in pop])
        X = boundary_handling(X, self.bounds[0, :], self.bounds[1, :])

        for i in range(len(pop)):
            X[i, self.id_i] = map(int, X[i, self.id_i])
            pop[i][idx] = X[i, :]
        return pop

    def recombine(self, id1, id2):
        p1 = self.pop_mu[id1]
        if id1 != id2:
            p2 = self.pop_mu[id2]
            # intermediate recombination for the mutation strengths
            p1[self._id_hyperpar] = (np.array(p1[self._id_hyperpar]) + \
                np.array(p2[self._id_hyperpar])) / 2
            # dominant recombination
            mask = randn(self.dim) > 0.5
            p1[mask] = p2[mask]
        return p1

    def select(self):
        pop = self.pop_mu + self.pop_lambda if self.plus_selection else self.pop_lambda
        fitness = np.r_[self.f_mu, self.f_lambda] if self.plus_selection else self.f_lambda
        
        fitness_rank = argsort(fitness)
        if not self.minimize:
            fitness_rank = fitness_rank[::-1]

        sel_id = fitness_rank[:self.mu_]
        self.pop_mu = pop[sel_id]
        self.f_mu = fitness[sel_id]

    def evaluate(self, pop):
        if not hasattr(pop[0], '__iter__'):
            pop = [pop]
        N = len(pop)
        f = np.zeros(N)
        for i, individual in enumerate(pop):
            var = individual[self._id_var]
            
            f[i] = np.sum(self.obj_func(var)) # in case a 1-length array is returned
            self.eval_count += 1
        return f

    def mutate(self, individual):
        if self.N_r:
            self._mutate_r(individual)
        if self.N_i:
            self._mutate_i(individual)
        if self.N_d:
            self._mutate_d(individual)
        return individual

    def _mutate_r(self, individual):
        sigma = np.array(individual[self._id_sigma])
        if len(self._id_sigma) == 1:
            sigma = sigma * exp(self.tau_r * randn())
        else:
            sigma = sigma * exp(self.tau_r * randn() + self.tau_p_r * randn(self.N_r))

        individual[self._id_sigma] = sigma
        individual[self.id_r] = np.array(individual[self.id_r]) + sigma * randn(self.N_r)

    def _mutate_i(self, individual):
        eta = np.array(individual[self._id_eta])
        if len(self._id_eta) == 1:
            eta = max(1, eta * exp(self.tau_i * randn()))
            p = 1 - (eta / self.N_i) / (1 + np.sqrt(1 + (eta / self.N_i) ** 2))
            individual[self.id_i] = np.array(individual[self.id_i]) + \
                geometric(p, self.N_i) - geometric(p, self.N_i)
        else:
            eta = eta * exp(self.tau_i * randn() + self.tau_p_i * randn(self.N_i))
            eta[eta > 1] = 1
            p = 1 - (eta / self.N_i) / (1 + np.sqrt(1 + (eta / self.N_i) ** 2))
            individual[self.id_i] = np.array(individual[self.id_i]) + \
                np.array([geometric(p_) - geometric(p_) for p_ in p])
        individual[self._id_eta] = eta

    def _mutate_d(self, individual):
        P = np.array(individual[self._id_p])
        P = 1 / (1 + (1 - P) / P * exp(-self.tau_d * randn()))
        individual[self._id_p] = boundary_handling(P, 1 / (3. * self.N_d), 0.5)[0].tolist()

        idx = np.nonzero(rand(self.N_d) < P)[0]
        for i in idx:
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

        if self.eval_count != 0:
            fitness = self.f_lambda
            # sigma = np.atleast_2d([__[self._id_sigma] for __ in self.pop_mu]) 
            # sigma_mean = np.mean(sigma, axis=0)
            
            # tolerance on fitness in history
            self.histfunval[mod(self.eval_count / self.lambda_ - 1, self.nbin)] = fitness[0]
            if mod(self.eval_count / self.lambda_, self.nbin) == 0 and \
                (max(self.histfunval) - min(self.histfunval)) < self.tolfun:
                    self.stop_dict['tolfun'] = True
            
            # flat fitness within the population
            if fitness[0] == fitness[int(min(ceil(.1 + self.lambda_ / 4.), self.mu_ - 1))]:
                self.stop_dict['flatfitness'] = True
            
            # TODO: implement more stop criteria
            # if any(sigma_mean < 1e-10) or any(sigma_mean > 1e10):
            #     self.stop_dict['sigma'] = True

            # if cond(self.C) > 1e14:
            #     if is_stop_on_warning:
            #         self.stop_dict['conditioncov'] = True
            #     else:
            #         self.flg_warning = True

            # # TolUPX
            # if any(sigma*sqrt(diagC)) > self.tolupx:
            #     if is_stop_on_warning:
            #         self.stop_dict['TolUPX'] = True
            #     else:
            #         self.flg_warning = True
            
        return any(self.stop_dict.values())

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
                self.pop_lambda[i] = self.mutate(individual)
            
            # TODO: the constraint handling method here will (by chance) turn really bad cadidates
            # (the one with huge sigmas) to good ones and hence making the step size explode
            self.keep_in_bound(self.pop_lambda)
            self.f_lambda = self.evaluate(self.pop_lambda)
            self.select()

            curr_best = self.pop_mu[0]
            xopt_, fopt_ = curr_best[self._id_var], self.f_mu[0]
            
            self.iter_count += 1

            if self._better(fopt_, self.fopt):
                self.xopt, self.fopt = xopt_, fopt_

            if self.verbose:
                print 'iteration ', self.iter_count + 1
                print curr_best[self._id_hyperpar], self.fopt

        self.stop_dict['funcalls'] = self.eval_count
        return self.xopt, self.fopt, self.stop_dict


if __name__ == '__main__':

    np.random.seed(1)
    # def fitness(x):
    #     x_r, x_i, x_d = np.array(x[:2]), x[2], x[3]
    #     if x_d == 'OK':
    #         tmp = 0
    #     else:
    #         tmp = 1
    #     return np.sum(x_r ** 2) + abs(x_i - 10) / 123. + tmp * 2
    
    def fitness(x):
        x_r = np.array(x[:2])
        return np.sum(x_r ** 2) 

    # x0 = [2, 1, 80, 'B']
    # bounds = [[-5, -5, -100], [5, 5, 100]]
    # levels = [['OK', 'A', 'B', 'C', 'D', 'E', 'F', 'G']]

    x0 = [2, 1]
    bounds = [[-5, -5], [5, 5]]

    opt = mies(x0, fitness, bounds, None, ['C', 'C'], 5e4, verbose=True)
    print opt.optimize()
