# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:44:21 2017

@author: wangronin
"""

import warnings
import numpy as np
from numpy import sqrt
from scipy.stats import norm

normcdf, normpdf = norm.cdf, norm.pdf

# TODO: turn this into a class
def EI(model, plugin, minimize=True):
    def _EI(X):
        # check on X should be handled in predict
        y_hat, mse = model.predict(X, eval_MSE=True)
        sigma = sqrt(mse)

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                diff = plugin - y_hat if minimize else y_hat - plugin
                value = diff * normcdf(diff / sigma) + sigma * normpdf(diff / sigma)
            except Warning:
                value = 0

        return value
    return _EI

def ei_dx(model, plugin=None):
    def __ei_dx(X):
        X = np.atleast_2d(X)
        X = X.T if X.shape[1] != model.X.shape[1] else X
        y = model.y

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