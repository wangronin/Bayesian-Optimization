#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:01:02 2017

@author: wangronin
"""

import pdb

import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from deap import benchmarks

from configurator import configurator

np.random.seed(666)

def objective(X):
    X = np.atleast_2d(X)
    return np.array([benchmarks.rastrigin(x)[0] for x in X])

dim = 2
n_init_sample = 500

x_lb = np.array([-5] * dim)
x_ub = np.array([5] * dim)

X = np.random.rand(n_init_sample, dim) * (x_ub - x_lb) + x_lb
y = objective(X)


def fitness(x):
    x_r, x_i, x_d = np.array(x[:2]), x[2], x[3]
    if x_d == 'OK':
        tmp = 0
    else:
        tmp = 1
    return np.sum(x_r ** 2) + abs(x_i - 10) / 123. + tmp * 2

def svm_from_cfg(cfg):
    """ Creates a SVM based on a configuration and evaluates it on the
    iris-dataset using cross-validation.

    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!

    Returns:
    --------
    A crossvalidated mean score for the svm on the loaded data-set.
    """
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
#    cfg = {k : cfg[k] for k in cfg if cfg[k]}
    # We translate boolean values:
#    cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
    # And for gamma, we set it to a fixed value or to "auto" (if used)
#    if "gamma" in cfg:
#        cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
#        cfg.pop("gamma_value", None)  # Remove "gamma_value"

    np.random.seed(1)

    x = np.array(cfg.values())

    # regr = svm.SVR(degree=2, max_iter=1e5, **cfg)

    # scores = cross_val_score(regr, X, y, cv=5, scoring='r2')
    return benchmarks.himmelblau(x)[0]

kernel = {'name' : "x1",
          'type' : 'R',
          'bounds': [-6, 6]}

kernel2 = {'name' : "x2",
          'type' : 'R',
          'bounds': [-6, 6]}

C = {'name' : "x3",
          'type' : 'I',
          'bounds': [-100, 100]}

gamma = {'name' : "x4",
          'type' : 'D',
          'levels': ['OK', 'E', 'F', 'G']}

conf_space = [kernel, kernel2]

# kernel = {'name' : "x1",
#           'type' : 'D',
#           'levels' : ["rbf", "sigmoid"]}

# C = {'name' : "C",
#           'type' : 'R',
#           'bounds': [0.001, 500]}

# gamma = {'name' : "gamma",
#           'type' : 'R',
#           'bounds': [0.0001, 5]}

# shrinking = {'name' : "shrinking",
#              'type' : 'D',
#              'levels': [True, False]}

# conf_space = [kernel, C, gamma, shrinking]

conf = configurator(conf_space, svm_from_cfg, 20, n_init_sample=10, minimize=True,
                    verbose=True)
conf.configure()
