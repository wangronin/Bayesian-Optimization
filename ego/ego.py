# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 12:31:07 2014

@author: wangronin


"""

import pdb
import numpy as np
from copy import deepcopy, copy

from pyDOE import lhs

from GaussianProcess import GaussianProcess_extra as GaussianProcess

from scipy.stats import norm
from numpy.random import rand
from numpy import array, sqrt, nonzero, inf

from .boundary_handling import boundary_handling
from .cma_es import cma_es
from scipy.optimize import fmin_l_bfgs_b

import warnings

normcdf, normpdf = norm.cdf, norm.pdf

class infill_criteria:

    def __init__(self):
        pass

def ei(model, plugin=None):

    def __ei(X):

        X = np.atleast_2d(X)

        X = X.T if X.shape[1] != model.X.shape[1] else X

        n_sample = X.shape[0]

        if True:
            #here you did de-standardization? Not needed with OWCK
            y = model.y # * np.std(model.y) + np.mean(model.y)
        else:
            y = model.y * model.y_std + model.y_mean

        fmin = np.min(y) if plugin is None else plugin

        y_pre = []
        mse = []
        for sample in X:
            y_sample, mse_sample = model.predict(sample, eval_MSE=True)
            y_pre.append(y_sample[0])
            mse.append(mse_sample[0])

        y_pre = np.array(y_pre)
        mse = np.array(mse)
        y_pre = y_pre.reshape(n_sample)
        mse = mse.reshape(n_sample)

        sigma = sqrt(mse)

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                value = (fmin - y_pre) * normcdf((fmin - y_pre) / sigma) + \
                    sigma * normpdf((fmin - y_pre) / sigma)
            except Warning:
                value = 0.

        return value

    return __ei


def ei_dx(model, plugin=None):

    def __ei_dx(X):

        X = np.atleast_2d(X)

        X = X.T if X.shape[1] != model.X.shape[1] else X

        if True:
            #here you did de-standardization? Not needed with OWCK
            y = model.y# * np.std(model.y) + np.mean(model.y)
        else:
            y = model.y * model.y_std + model.y_mean

        fmin = np.min(y) if plugin is None else plugin

        y, sd2 = model.predict(X, eval_MSE=True)
        sd = np.sqrt(sd2)

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                y_dx, sd2_dx = model.gradient(X)
                sd_dx = sd2_dx / (2. * sd)

                xcr = (fmin - y) / sd
                xcr_prob, xcr_dens = normcdf(xcr), normpdf(xcr)

                grad = -y_dx * xcr_prob + sd_dx * xcr_dens

            except Warning:
                grad = np.zeros((X.shape[1], 1))

        return grad

    return __ei_dx


def pi(model, dim, plugin=None):

    def __pi(X):

        X = np.atleast_2d(X)

        X = X.T if X.shape[0] == dim else X

        y = model.y * model.y_std + model.y_mean

        fmin = np.min(y) if plugin is None else plugin

        y_pre, mse = model.predict(X, eval_MSE=True)
        sigma = sqrt(mse)

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                value = normcdf((fmin - y_pre) / sigma)
            except Warning:
                value = 0.

        return value

    return __pi


def pi_dx(model, plugin=None):

    def __pi_dx(X):

        if True:
            #here you did de-standardization? Not needed with OWCK
            y = model.y# * np.std(model.y) + np.mean(model.y)
        else:
            y = model.y * model.y_std + model.y_mean

        fmin = np.min(y) if plugin is None else plugin

        y, sd2 = model.predict(X, eval_MSE=True)
        sd = np.sqrt(sd2)

        y_dx, sd2_dx = model.gradient(X)
        sd_dx = sd2_dx / (2. * sd)

        xcr = (fmin - y) / sd
        xcr_dens = normpdf(xcr)

        grad = -(y_dx + xcr * sd_dx) * xcr_dens / sd

        return grad

    return __pi_dx


class ego:

    def __init__(self, dim, fitness, model, n_step, lb, ub,
                 doe_size=None,
                 doe_method='LHS',
                 criterion='EI',
                 is_minimize=True,
                 solver='BFGS',
                 verbose=False,
                 random_seed=999,
                 n_restart=None,
                 ):

        assert isinstance(dim, int)

        # TODO: add an parameter to control the how frequently the model is re-estimated
        self.is_minimize = is_minimize
        self.verbose = verbose
        self.dim = int(dim)
        self.n_step = n_step     # number of steps
        self.random_seed = random_seed
        self.itercount = 0
        self.current_y = np.inf
        self.best_y = np.inf
        self.y_hist = []

        # if the model has been properly update in each iteration
        self.is_updated = False

        self.sd_threshold = 0.002    # minimal allowed variance value
        self.non_explorative_count = 0

        # set the new seed
#        random.seed(self.random_seed)
#        np.random.seed(self.random_seed)

        # TODO: implement UCB
        if criterion not in ['EI', 'PI', 'UCB']:
            raise Exception('Unsupported in-fill criterion!')
        else:
            self.criterion = criterion

        # TODO: callable solvers
        if solver not in ['CMA', 'BFGS', 'CMA-tree', 'BFGS-tree']:
            raise Exception('Unsupported solver for in-fill criterion!')
        else:
            self.solver = solver

        if hasattr(fitness, '__call__'):
            self.fitness = fitness if is_minimize else lambda x: -fitness(x)
        else:
            raise Exception('fitness function is not callable!')

        if len(lb) != self.dim or len(ub) != self.dim:
            raise Exception('Length of bounds does not match dimension!')

        x_lb = np.atleast_2d(lb)
        x_ub = np.atleast_2d(ub)

        self.x_lb = x_lb if x_lb.shape[0] == 1 else x_lb.T
        self.x_ub = x_ub if x_ub.shape[0] == 1 else x_ub.T

        self.model = deepcopy(model)
        if hasattr(self.model, 'X'):    # given fitted Kriging model
            if not isinstance(model, GaussianProcess):
                raise Exception('Unsupported Model class!')

            # make a copy, to be safe :)
            self.X = copy(self.model.X)
            self.y = copy(self.model.y)
            self.doe_size = self.X.shape[0]

            if self.y.ndim != 1:
                self.y = self.y.flatten()

        else: # otherwise the model will be trained
            self.doe_method = doe_method
            self.doe_size = int(doe_size)
            self.X = self.DOE(self.doe_size)
            self.y = self.evaluation(self.X)

            if self.verbose:
                print 'creating the GPR model on DOE samples...'
            self.update_model(self.X, self.y, re_estimation=True)

        # restart upper bound for L-BFGS-B algorithm
        # TODO: using benchmark to select a good setting here
        self.n_restarts_optimizer = int(30 * self.dim) if n_restart is None else n_restart
        self.eval_budget_criterion = 1e3*self.dim
        self.wait_iter = 10

    def evaluation(self, X):
        y = array([self.fitness(x) for x in X]).flatten()

        return y

    def DOE(self, doe_size, method='LHS'):
        # design of experiments
        doe_method = 'uniform' if doe_size == 1 else self.doe_method

        if doe_method == 'LHS':
            X = lhs(self.dim, samples=doe_size, criterion='cm') \
            	* (self.x_ub - self.x_lb) + self.x_lb
        elif doe_method == 'uniform':
            X = rand(doe_size, self.dim) * (self.x_ub - self.x_lb) + self.x_lb

        return X

    def update_model(self, X=None, y=None, re_estimation=False):
        if re_estimation:
            if self.verbose:
                print 'model (re)-fitting...'
            self.model.fit(self.X, self.y)
        else:
            if self.verbose:
                print 'model updating...'
            self.model.update(X, y)

        self.is_updated = True

    def optimize(self):
        # The main interface
        # TODO: add more stop criteria...
        for i in range(self.n_step):
            _ = self.step()

        new_X, new_y = self.X[self.doe_size:, :], self.y[self.doe_size:]
        return self.best_x, self.best_y, new_X, new_y, self.y_hist

    def step(self):
        self.itercount += 1
        self.is_updated = False

        while True:
            # maximization of EI
            new_x, ei_value, _, __ = self.max_criterion()

            new_x = boundary_handling(new_x.reshape(-1, 1), self.x_lb, self.x_ub)
            new_x = new_x.T

            # check the re-estimation criteria
            re_estimate = self.check_re_estimation(new_x, ei_value)

            # check for potential duplications
            new_x = self.__remove_duplicate(new_x)

            # if no new design site found, re-estimate the parameters immediately
            if len(new_x) == 0:
                if not self.is_updated:
                    # Duplication are commonly encountered in the 'corner'
                    self.update_model(self.X, self.y, re_estimation=True)
                else:

                    # getting duplcations by this is of 0 measure...
                    warnings.warn('iteration {}: duplicated solution found \
                                  by optimization! New points is taken from random \
                                  design'.format(self.itercount))
                    new_x = self.DOE(1, method='uniform')
                    break
            else:
                break

        # evaluation and the append the new data point
        new_y = self.evaluation(new_x)[0]
        self.X = np.r_[self.X, new_x]
        self.y = np.r_[self.y, new_y]

        self.current_y = new_y
        self.best_y = np.min(self.y)
        self.best_x = self.X[nonzero(self.y == np.min(self.y))[0], ]
        self.y_hist.append(self.best_y)

        if self.verbose:
            print 'iteration {}, best fitness found {}'.format(self.itercount,
                             self.best_y)

        # update the model: the model parameters are not necessarily re-estimated
        self.update_model(new_x, new_y, re_estimation=True)

        return self.best_x, self.best_y, new_x, new_y, ei_value

    def check_sd2(self, x):

        _, sd2 = self.model.predict(x, eval_MSE=True)
        if np.sqrt(sd2) <= self.sd_threshold:
            self.non_explorative_count += 1
        else:
            self.non_explorative_count = 0

    def check_re_estimation(self, new_x, ei_value):

        self.check_sd2(new_x)

        is_re_estimate = False
        if self.non_explorative_count == 5:
            is_re_estimate = True
            self.non_explorative_count = 0

        is_re_estimate = is_re_estimate or np.isclose(ei_value, 0)
        return is_re_estimate

    def max_criterion(self, plugin=None):

        if plugin is None:
            plugin = np.min(self.y)

        xopt_list, ei_list = {}, {}

        # settings
        fopt = np.inf
        eval_budget = self.eval_budget_criterion
        EI, EI_dx = ei(self.model), ei_dx(self.model)
        def obj_func(x):
            return -EI(x), -EI_dx(x)

        if self.verbose:
            print 'The chosen optimizer is {}'.format(self.solver)

        if self.solver == 'BFGS':      # quasi-newton method with restarts

            bounds = np.c_[self.x_lb.T, self.x_ub.T]

            if not np.isfinite(bounds).all() and self.n_restarts_optimizer > 1:
                raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")

            # L-BFGS-B algorithm with restarts
            c = 0
            for iteration in range(self.n_restarts_optimizer):
                x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
                xopt_, fopt_, stop_info = fmin_l_bfgs_b(obj_func, x0, pgtol=1e-8,
                                                        factr=1e6, bounds=bounds,
                                                        maxfun=eval_budget)

                if stop_info["warnflag"] != 0 and self.verbose:
                    warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                          " state: %s" % stop_info)

                if fopt_ < fopt:
                    xopt, fopt = xopt_, fopt_
                    if self.verbose:
                        print 'iteration: ', iteration+1, stop_info['funcalls'], fopt_
                    c = 0
                else:
                    c += 1

                eval_budget -= stop_info['funcalls']
                if eval_budget <= 0 or c >= self.wait_iter:
                    break

            fopt = -fopt

        elif self.solver == 'BFGS-tree':      # quasi-newton method with restarts

            leaf_bounds = self.get_tree_boundary()
            total_budget = 1e3*self.dim
            total_measure = np.prod(self.x_ub - self.x_lb)

            # TODO: this part should be parallelizable
            fopt = inf
            for leaf_node, bounds in leaf_bounds.iteritems():

                x_lb, x_ub = zip(*[bounds[i] for i in bounds.keys()])
                x_lb = np.atleast_2d(x_lb)
                x_ub = np.atleast_2d(x_ub)
                bounds = np.c_[x_lb.T, x_ub.T]

                measure = np.prod(x_ub - x_lb)

                # TODO: ceiling or floor?
                eval_budget = int(measure / total_measure * total_budget)

                xopt_, fopt_ = self._BFGS_restart(obj_func, bounds, eval_budget)

                xopt_list[leaf_node] = xopt_
                ei_list[leaf_node] = -fopt_
                if fopt_ < fopt:
                    xopt, fopt = xopt_, fopt_

            fopt = -fopt

        elif self.solver == 'CMA':        # CMA-ES solver
            # Algorithm parameters
            opt = { \
                'sigma_init': 0.25 * np.max(self.x_ub - self.x_lb),
                'eval_budget': 1e3*self.dim,
                'f_target': inf,
                'lb': self.x_lb.T,
                'ub': self.x_ub.T \
                }

            init_wcm = rand(1, self.dim) * (self.x_ub - self.x_lb) + \
                self.x_lb

            fitnessfunc = ei(self.model)
            optimizer = cma_es(self.dim, init_wcm, fitnessfunc, opt,
                               is_minimize=False)
            xopt, fopt, evalcount, _ = optimizer.optimize()
            xopt = xopt.T
            fopt = -fopt

        # CMA-ES with Decision tree search
        elif self.solver == 'CMA-tree':
            leaf_bounds = self.get_tree_boundary()
            total_budget = 1e3*self.dim
            total_measure = np.prod(self.x_ub - self.x_lb)

            # TODO: this part should be parallelizable
            fopt = inf
            for leaf_node, bounds in leaf_bounds.iteritems():

                x_lb, x_ub = zip(*[bounds[i] for i in bounds.keys()])
                x_lb = np.atleast_2d(x_lb)
                x_ub = np.atleast_2d(x_ub)

                measure = np.prod(x_ub - x_lb)
                eval_budget = int(measure / total_measure * total_budget)

                # Algorithm parameters
                opt = {
                       'sigma_init': 0.25 * np.max(x_ub - x_lb),
                       'eval_budget': eval_budget,
                       'f_target': inf,
                       'lb': x_lb.T,
                       'ub': x_ub.T
                       }

                init_wcm = rand(1, self.dim) * (x_ub - x_lb) + x_lb
                fitnessfunc = ei(self.model)

                optimizer = cma_es(self.dim, init_wcm, fitnessfunc, opt,
                                   is_minimize=False)
                xopt_, fopt_, evalcount, _ = optimizer.optimize()

                xopt_list[leaf_node] = xopt_
                ei_list[leaf_node] = -fopt_
                if fopt_ < fopt:
                    xopt, fopt = xopt_.T, fopt_

            fopt = -fopt

        elif self.solver == 'pruning':
            pass

        elif self.solver == 'BB':   # Branch and Bound solver
            # TODO: implement this for Gaussian kernel
            pass
        elif self.solver == 'genoud':    # Genetic optimization with derivatives
            # TODO: somehow get it working from R package and test its performance ag
            pass

        if self.verbose:
            print 'iteration {}, EI: {}'.format(self.itercount, fopt)

        return xopt, fopt, xopt_list, ei_list

    def _BFGS_restart(self, obj_func, bounds, eval_budget):

        fopt = np.inf
        # L-BFGS-B algorithm with restarts
        for iteration in range(self.n_restarts_optimizer):

            x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])

            # TODO: expose the parameters of BFGS to the users
            xopt_, fopt_, convergence_dict = \
                fmin_l_bfgs_b(obj_func, x0, pgtol=1e-30, factr=1e2,
                              bounds=bounds, maxfun=eval_budget)

            if convergence_dict["warnflag"] != 0 and self.verbose:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                      " state: %s" % convergence_dict)

            if fopt_ < fopt:
                if self.verbose:
                    print 'iteration: ', iteration+1, convergence_dict['funcalls'], fopt_
                xopt, fopt = xopt_, fopt_

            eval_budget -= convergence_dict['funcalls']
            if eval_budget <= 0:
                break

        return xopt, fopt

    def __remove_duplicate(self, new_x):

        # TODO: show a warning here
        new_x = np.atleast_2d(new_x)
        samples = []
        for x in new_x:
            if not any(np.all(np.isclose(self.X, x), axis=1)):
                samples.append(x)

        return array(samples)

    # TODO: this part should be moved the OWCK
    def get_tree_boundary(self):
        """
        Get the variable boundary for each leaf node
        """

        def recurse_new(node, decision_path, bounds):
            """
            Traverse the decision tree in preoder to retrieve the
            decision path for leaf nodes
            """
            decision_path_ = deepcopy(decision_path)
            if ('left' in node.keys()):
                left_child = node['left']
                right_child = node['right']
                split_feature = node['index']
                threshold = node['value']
            else:
                #leaf node
                leaf_node_path[node['id']] = decision_path_
                leaf_bounds[node['id']] = bounds
                return

            decision_path_.append(node['node_id']) # pre-order visit
            __ = bounds[split_feature]

            # proceed to the left child
            bounds_left = deepcopy(bounds)
            bounds_left[split_feature] = update_bounds(__, leq=threshold)
            recurse_new(left_child, decision_path_, bounds_left)
            # proceed to the right child
            bounds_right = deepcopy(bounds)
            bounds_right[split_feature] = update_bounds(__, geq=threshold)
            recurse_new(right_child, decision_path_, bounds_right)

        def recurse(tree, node_id, decision_path, bounds):
            """
            Traverse the decision tree in preoder to retrieve the
            decision path for leaf nodes
            """
            decision_path_ = deepcopy(decision_path)
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            split_feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            if split_feature == -2: # leaf node
                leaf_node_path[node_id] = decision_path_
                leaf_bounds[node_id] = bounds
                return
            else:
                decision_path_.append(node_id) # pre-order visit
                __ = bounds[split_feature]

                # proceed to the left child
                bounds_left = deepcopy(bounds)
                bounds_left[split_feature] = update_bounds(__, leq=threshold)
                recurse(tree, left_child, decision_path_, bounds_left)
                # proceed to the right child
                bounds_right = deepcopy(bounds)
                bounds_right[split_feature] = update_bounds(__, geq=threshold)
                recurse(tree, right_child, decision_path_, bounds_right)

        def update_bounds(bounds, leq=None, geq=None):
            lb, ub = bounds
            if leq is not None:
                if leq <= lb:
                    raise Exception
                ub = leq if leq < ub else ub

            if geq is not None:
                if geq >= ub:
                    raise Exception
                lb = geq if geq > lb else lb

            return [lb, ub]

        leaf_node_path = {}
        leaf_bounds = {}
        tree = self.model.clusterer.tree_

        recurse_new(tree, [],
                {i: [self.x_lb[0, i], self.x_ub[0, i]] for i in range(self.dim)})

        return leaf_bounds


