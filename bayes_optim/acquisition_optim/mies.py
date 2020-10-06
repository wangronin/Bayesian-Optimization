# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:10:18 2017

@author: wangronin
"""
import numpy as np
from numpy import exp, nonzero, argsort, ceil, zeros, mod
from numpy.random import randint, rand, randn, geometric

from ..misc import handle_box_constraint
from ..base import Solution
from ..utils import dynamic_penalty
from ..SearchSpace import ContinuousSpace, OrdinalSpace, NominalSpace

# TODO: test more contraint handling methods
# TODO: improve efficiency, e.g. compile it with cython
# TODO: try to use advanced python default parameter
# TODO: re-structure the code and add `parallel_obj_fun`

class MIES(object):
    def __init__(
        self,
        search_space,
        obj_func,
        eq_func=None,
        ineq_func=None,
        x0=None,
        ftarget=None,
        max_eval=np.inf,
        minimize=True,
        elitism=False,
        mu_=4,
        lambda_=10,
        sigma0=None,
        eta0=None,
        P0=None,
        verbose=False,
        eval_type='list'):

        self.mu_ = mu_
        self.lambda_ = lambda_
        self.eval_count = 0
        self.iter_count = 0
        self.minimize = minimize
        self.obj_func = obj_func
        self.eq_func = eq_func
        self.ineq_func = ineq_func
        self.stop_dict = {}
        self.verbose = verbose
        self.max_eval = max_eval
        self.ftarget = ftarget
        self.elitism = False
        self._penalty_func = dynamic_penalty
        self._eval_type = eval_type

        self._space = search_space
        self.var_names = self._space.var_name
        self.param_type = self._space.var_type

        # index of each type of variables in the dataframe
        self.id_r = self._space.id_C       # index of continuous variable
        self.id_i = self._space.id_O       # index of integer variable
        self.id_d = self._space.id_N       # index of categorical variable

        # the number of variables per each type
        self.N_r = len(self.id_r)
        self.N_i = len(self.id_i)
        self.N_d = len(self.id_d)
        self.dim = self.N_r + self.N_i + self.N_d

        # by default, we use individual step sizes for continuous and
        # integer variables and global strength for the nominal variables
        self.N_p = min(self.N_d, int(1))

        # total length of the solution vector
        self._len = self.dim + self.N_r + self.N_i + self.N_p

        # unpack interval bounds
        self.bounds_r = np.asarray([self._space.bounds[_] for _ in self.id_r])
        self.bounds_i = np.asarray([self._space.bounds[_] for _ in self.id_i])
        self.bounds_d = np.asarray([self._space.bounds[_] for _ in self.id_d])
        self._check_bounds(self.bounds_r)
        self._check_bounds(self.bounds_i)

        # step default step-sizes/mutation strength
        par_name = []
        if sigma0 is None and self.N_r:
            sigma0 = 0.05 * (self.bounds_r[:, 1] - self.bounds_r[:, 0])
            par_name += ['sigma' + str(_) for _ in range(self.N_r)]
        if eta0 is None and self.N_i:
            eta0 = 0.05 * (self.bounds_i[:, 1] - self.bounds_i[:, 0])
            par_name += ['eta' + str(_) for _ in range(self.N_i)]
        if P0 is None and self.N_d:
            P0 = 1. / self.N_d
            par_name += ['P' + str(_) for _ in range(self.N_p)]

        # column indices: used for slicing
        self._id_var = np.arange(self.dim)
        self._id_sigma = np.arange(self.N_r) + len(self._id_var)
        self._id_eta = np.arange(self.N_i) + len(self._id_var) + len(self._id_sigma)
        self._id_p = np.arange(self.N_p) + len(self._id_var) + len(self._id_sigma) \
            + len(self._id_eta)
        self._id_hyperpar = np.arange(self.dim, self._len)

         # initialize the populations
        if x0 is not None:                         # given x0
            par = []
            if self.N_r:
                par += [sigma0]
            if self.N_i:
                par += [eta0]
            if self.N_p:
                par += [P0] * self.N_p

            self.pop = Solution(np.tile(np.r_[x0, par], (self.mu_, 1)),
                                var_name=self.var_names + par_name,
                                verbose=self.verbose)
            fitness0 = self.evaluate(self.pop[0])
            self.fitness = np.repeat(fitness0, self.mu_)
            self.xopt = x0
            self.fopt = sum(fitness0)
        else:                                      # uniform sampling
            x = np.asarray(self._space.sampling(self.mu_), dtype='object')

            par = []
            if self.N_r:
                par += [np.tile(sigma0, (self.mu_, 1))]
            if self.N_i:
                par += [np.tile(eta0, (self.mu_, 1))]
            if self.N_p:
                par += [np.tile([P0] * self.N_p, (self.mu_, 1))]

            par = np.concatenate(par, axis=1)
            x = np.c_[x, par].tolist()

            self.pop = Solution(x, var_name=self.var_names + par_name, verbose=self.verbose)
            self.fitness = self.evaluate(self.pop)
            self.fopt = min(self.fitness) if self.minimize else max(self.fitness)
            _ = np.nonzero(self.fopt == self.fitness)[0][0]
            self.xopt = self.pop[_, self._id_var]

        self.offspring = self.pop[0] * self.lambda_
        self.f_offspring = np.repeat(self.fitness[0], self.lambda_)
        self._set_hyperparameter()

        # stopping criteria
        self.tolfun = 1e-5
        self.nbin = int(3 + ceil(30. * self.dim / self.lambda_))
        self.histfunval = zeros(self.nbin)

    def _check_bounds(self, bounds):
        if len(bounds) == 0:
            return
        if any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError('lower bounds must be smaller than upper bounds')

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

    def recombine(self, id1, id2):
        p1 = self.pop[id1].copy()  # IMPORTANT: make a copy
        if id1 != id2:
            p2 = self.pop[id2]
            # intermediate recombination for the mutation strengths
            p1[self._id_hyperpar] = (np.array(p1[self._id_hyperpar]) + \
                np.array(p2[self._id_hyperpar])) / 2

            # dominant recombination for solution parameters
            _, = np.nonzero(randn(self.dim) > 0.5)
            p1[_] = p2[_]
        return p1

    def select(self):
        pop = self.pop + self.offspring if self.elitism else self.offspring
        fitness = np.r_[self.fitness, self.f_offspring] if self.elitism else self.f_offspring
        rank = argsort(fitness)

        if not self.minimize:
            rank = rank[::-1]

        _ = rank[:self.mu_]
        self.pop = pop[_]
        self.fitness = fitness[_]

    def evaluate(self, pop, return_penalized=True):
        X = pop[:, self._id_var]
        if self._eval_type == 'list':
            X = X.tolist()
        elif self._eval_type == 'dict':
            X = X.to_dict()

        if len(pop.shape) == 1:  # one solution
            # make `np.array` here is not needed
            pop.fitness = np.array(self.obj_func(X))
        else:                    # for a population
            pop.fitness = self.obj_func(X)
            # pop.fitness = [self.obj_func(x) for x in X]

        self.eval_count += pop.N
        _penalized_fitness = pop.fitness + \
            self._penalty_func(
                X, self.iter_count + 1,
                self.eq_func, self.ineq_func,
                minimize=self.minimize
            )
        return (_penalized_fitness if return_penalized else pop.fitness)

    def mutate(self, X):
        if self.N_r:
            self._mutate_r(X)
        if self.N_i:
            self._mutate_i(X)
        if self.N_d:
            self._mutate_d(X)
        return X

    def _mutate_r(self, X):
        n_point = X.shape[0]
        sigma = np.array(X[:, self._id_sigma], dtype=float).reshape(n_point, -1)
        X_ = np.array(X[:, self.id_r], dtype=float)

        # mutate step-sizes
        if len(self._id_sigma) == 1:
            sigma *= exp(self.tau_r * randn(n_point, 1))
        else:
            sigma *= exp(self.tau_r * randn(n_point, 1) + \
                self.tau_p_r * randn(n_point, self.N_r))

        # Gaussian mutation
        R = randn(n_point, self.N_r)
        _X = X_ + sigma * R

        # Interval Bounds Treatment
        _X = handle_box_constraint(_X, self.bounds_r[:, 0], self.bounds_r[:, 1])

        # TODO: check if this interval boundary handling works with penalty functions
        # the constraint handling method will (by chance) turn really bad cadidates
        # (the one with huge sigmas) to good ones and hence making the step size explode
        # Repair the step-size if `_X` is out of bounds
        if 1 < 2:
            X[:, self._id_sigma] = np.abs((_X - X_) / R)
        else:
            X[:, self._id_sigma] = sigma
        X[:, self.id_r] = _X

    def _mutate_i(self, X):
        n_point = X.shape[0]
        eta = np.array(X[:, self._id_eta], dtype=float).reshape(n_point, -1)
        X_ = np.array(X[:, self.id_i], dtype=int)

        if len(self._id_eta) == 1:
            eta *= exp(self.tau_i * randn(n_point, 1))
        else:
            eta *= exp(self.tau_i * randn(n_point, 1) + \
                self.tau_p_i * randn(n_point, self.N_i))

        eta[eta > 1] = 1
        p = 1 - (eta / self.N_i) / (1 + np.sqrt(1 + (eta / self.N_i) ** 2.))
        _X = X_ + geometric(p) - geometric(p)

        # Interval Bounds Treatment
        _X = handle_box_constraint(_X, self.bounds_i[:, 0], self.bounds_i[:, 1])
        _X = _X.astype(int)

        X[:, self.id_i] = _X
        X[:, self._id_eta] = eta

    def _mutate_d(self, X):
        n_point = X.shape[0]
        P = np.array(X[:, self._id_p], dtype=float).reshape(n_point, -1)

        #  Unbiased mutation on the mutation probability
        P = 1. / (1. + (1. - P) / P * exp(-1 * self.tau_d * randn(n_point, 1)))
        X[:, self._id_p] = handle_box_constraint(P, 1. / (3. * self.N_d), 0.5)

        idx = rand(n_point, self.N_d) < P
        for i in range(self.N_d):
            _ = idx[:, i]
            X[_, self.id_d[i]] = np.random.choice(self.bounds_d[i], sum(_))

    def stop(self):
        if self.eval_count > self.max_eval:
            self.stop_dict['max_eval'] = True

        if self.eval_count != 0 and self.iter_count != 0:
            fitness = self.f_offspring
            # sigma = np.atleast_2d([__[self._id_sigma] for __ in self.pop])
            # sigma_mean = np.mean(sigma, axis=0)

            # tolerance on fitness in history
            self.histfunval[int(mod(self.eval_count / self.lambda_ - 1, self.nbin))] = fitness[0]
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

    def _better(self, f1, f2):
        return f1 < f2 if self.minimize else f1 > f2

    def optimize(self):
        while not self.stop():

            for i in range(self.lambda_):
                p1, p2 = randint(0, self.mu_), randint(0, self.mu_)
                self.offspring[i] = self.recombine(p1, p2)

            self.offspring = self.mutate(self.offspring)

            # NOTE: `self.fitness` and `self.f_offspring` are penalized function values
            self.f_offspring[:] = self.evaluate(self.offspring)
            self.select()

            curr_best = self.pop[0]
            xopt_, fopt_ = curr_best[self._id_var], self.fitness[0]

            if self._better(fopt_, self.fopt):
                self.xopt, self.fopt = xopt_, fopt_
            self.iter_count += 1

            if self.verbose:
                print('iteration {}, fopt: {}'.format(self.iter_count + 1, self.fopt))
                print(self.xopt)

        self.stop_dict['funcalls'] = self.eval_count
        return self.xopt.tolist(), self.fopt, self.stop_dict