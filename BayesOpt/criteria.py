# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:44:21 2017

@author: wangronin
"""

import pdb
import warnings
import numpy as np
from numpy import sqrt
from scipy.stats import norm

normcdf, normpdf = norm.cdf, norm.pdf

class Criteria(object):
    def __init__(self, model, plugin=None, minimize=True):
        assert hasattr(model, 'predict')
        self.model = model
        self.plugin = plugin 
        self.minimize = minimize
    
    def __call__(self, X):
        pass

class EI(Criteria):
    def __call__(self, X, dx=False):
        X = np.atleast_2d(X)
        y_hat, sd2 = self.model.predict(X, eval_MSE=True)
        sd = sqrt(sd2)

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                # TODO: I have save xcr_ becasue xcr * sd != xcr_ numerically
                # find out the cause of such an error, probably representation error...
                xcr_ = self.plugin - y_hat if self.minimize else y_hat - self.plugin
                xcr = xcr_ / sd
                xcr_prob, xcr_dens = normcdf(xcr), normpdf(xcr)
                value = xcr_ * xcr_prob + sd * xcr_dens
            except Warning: # in case of numerical errors
                # TODO: find out which warning is generated and remove try...except
                value = 0
        if dx:
            assert hasattr(self.model, 'gradient')
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    y_dx, sd2_dx = self.model.gradient(X)
                    sd_dx = sd2_dx / (2. * sd)
                    grad = -y_dx * xcr_prob + sd_dx * xcr_dens
                except Warning:
                    grad = np.zeros((X.shape[1], 1))
            return value, grad 
        return value

# TODO: implement infill_criteria for noisy functions and MGF-based ceiterion
class PI(object):
    def __call__(self, X):
        pass

class GEI(object):
    def __call__(self, X):
        pass

class UCB(object):
    def __call__(self, X):
        pass