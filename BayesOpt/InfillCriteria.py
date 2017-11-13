# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:44:21 2017

@author: wangronin
"""

import pdb

import warnings
import numpy as np
from numpy import sqrt, exp, pi
from scipy.stats import norm
from abc import ABCMeta, abstractmethod

# warnings.filterwarnings("error")

# TODO: perphas also enable acquisition function engineering here?
# meaning the combination of the acquisition functions
class InfillCriteria:
    __metaclass__ = ABCMeta
    def __init__(self, model, plugin=None, minimize=True):
        assert hasattr(model, 'predict')
        self.model = model
        self.minimize = minimize
        # change maximization problem to minimization
        self.plugin = plugin if self.minimize else -plugin
        if self.plugin is None:
            self.plugin = np.min(model.y) if minimize else -np.max(self.model.y)
    
    @abstractmethod
    def __call__(self, X):
        raise NotImplementedError

    def _predict(self, X):
        y_hat, sd2 = self.model.predict(X, eval_MSE=True)
        sd = sqrt(sd2)
        if not self.minimize:
            y_hat = -y_hat
        return y_hat, sd

    def _gradient(self, X):
        y_dx, sd2_dx = self.model.gradient(X)
        sd_dx = sd2_dx / (2. * sd)
        if not self.minimize:
            y_dx = -y_dx
        return y_dx, sd_dx

    def check_X(self, X):
        """Keep input as '2D' object 
        """
        return [X] if not hasattr(X[0], '__iter__') else X

# TODO: test UCB implementation
class UCB(InfillCriteria):
    """
    Upper Confidence Bound 
    """
    def __init__(self, model, plugin=None, minimize=True, alpha=1e-10):
        super(EpsilonPI, self).__init__(model, plugin, minimize)
        self.alpha = alpha

    def __call__(self, X, dx=False):
        X = self.check_X(X)
        y_hat, sd = self._predict(X)

        try:
            f_value = y_hat + self.alpha * sd
        except Exception: # in case of numerical errors
            f_value = 0

        if dx:
            y_dx, sd_dx = self._gradient(X)
            try:
                f_dx = y_dx + self.alpha * sd_dx
            except Exception:
                f_dx = np.zeros((len(X[0]), 1))
            return f_value, f_dx 
        return f_value

class EI(InfillCriteria):
    """
    Expected Improvement
    """
    def __call__(self, X, dx=False):
        X = self.check_X(X)
        y_hat, sd = self._predict(X)

        try:
            # TODO: I have save xcr_ becasue xcr * sd != xcr_ numerically
            # find out the cause of such an error, probably representation error...
            xcr_ = self.plugin - y_hat
            xcr = xcr_ / sd
            xcr_prob, xcr_dens = norm.cdf(xcr), norm.pdf(xcr)
            f_value = xcr_ * xcr_prob + sd * xcr_dens
        except Exception: # in case of numerical errors
            f_value = 0

        if dx:
            y_dx, sd_dx = self._gradient(X)
            try:
                f_dx = -y_dx * xcr_prob + sd_dx * xcr_dens
            except Exception:
                f_dx = np.zeros((len(X[0]), 1))
            return f_value, f_dx 
        return f_value

class EpsilonPI(InfillCriteria):
    """
    epsilon-Probability of Improvement
    # TODO: verify the implementation
    """
    def __init__(self, model, plugin=None, minimize=True, epsilon=1e-10):
        super(EpsilonPI, self).__init__(model, plugin, minimize)
        self.epsilon = epsilon

    def __call__(self, X, dx=False):
        X = self.check_X(X)
        y_hat, sd = self._predict(X)

        coeff = 1 - self.epsilon if y_hat > 0 else (1 + self.epsilon)

        try:
            xcr_ = self.plugin - coeff * y_hat 
            xcr = xcr_ / sd
            f_value = norm.cdf(xcr)
        except Exception:
            f_value = 0.

        if dx:
            y_dx, sd_dx = self._gradient(X)
            try:
                f_dx = -(coeff * y_dx + xcr * sd_dx) * norm.pdf(xcr) / sd
            except Exception:
                f_dx = np.zeros((len(X[0]), 1))
            return f_value, f_dx 
        return f_value

class PI(EpsilonPI):
    """
    Probability of Improvement
    """
    def __init__(self, model, plugin=None, minimize=True):
        super(PI, self).__init__(model, plugin, minimize, epsilon=0)

class MGFI(InfillCriteria):
    """
    Moment-Generating Function of Improvement 
    My new acquisition function proposed in SMC'17 paper
    """
    def __init__(self, model, plugin=None, minimize=True, t=1):
        super(MGFI, self).__init__(model, plugin, minimize)
        self.t = t

    def __call__(self, X, dx=False):
        X = self.check_X(X)
        y_hat, sd = self._predict(X)

        y_hat_p = y_hat - self.t * sd ** 2.
        beta_p = (self.plugin - y_hat_p) / sd
        term = self.t * (self.plugin - y_hat - 1)
        value = norm.cdf(beta_p) * exp(term + self.t ** 2. * sd ** 2. / 2.)

        if np.isinf(value):
            value = 0.

        # TODO: implement this
        if dx:
            y_dx, sd_dx = self._gradient(X)
        return value
        
class GEI(InfillCriteria):
    """
    Generalized Expected Improvement 
    """
    def __init__(self, model, plugin=None, minimize=True, g=1):
        super(GEI, self).__init__(model, plugin, minimize)
        self.g = g

    def __call__(self, X, dx=False):
        pass

if __name__ == '__main__':

    # TODO: test for the gradient of Infill-Criteria
    # goes to unittest
    from GaussianProcess.trend import linear_trend
    from GaussianProcess import GaussianProcess
    from GaussianProcess.utils import plot_contour_gradient
    import matplotlib.pyplot as plt

    from deap import benchmarks

    np.random.seed(123)
    plt.ioff()
    fig_width = 21.5
    fig_height = fig_width / 3.2

    # def branin(xx, a=1, b=5.1/(4*pi**2.), c=5/pi, r=6, s=10, t=1/(8*pi)):
    #     x1, x2 = xx[:, 0], xx[:, 1]
    #     term1 = a * (x2 - b * x1 ** 2. + c * x1 - r) ** 2.
    #     term2 = s * (1 - t) * np.cos(x1)
    #     y = term1 + term2 + s
    #     return y
    
    def fitness(X):
        X = np.atleast_2d(X)
        return np.array([benchmarks.schwefel(x)[0] for x in X]) \
            + np.sqrt(noise_var) * np.random.randn(X.shape[0])

    dim = 2
    n_init_sample = 70
    noise_var = 0.1

    x_lb = np.array([-5] * dim)
    x_ub = np.array([5] * dim)

    X = np.random.rand(n_init_sample, dim) * (x_ub - x_lb) + x_lb
    y = fitness(X)

    thetaL = 1e-5 * (x_ub - x_lb) * np.ones(dim)
    thetaU = 10 * (x_ub - x_lb) * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

    mean = linear_trend(dim, beta=None)
    model = GaussianProcess(mean=mean,
                            corr='matern',
                            theta0=theta0,
                            thetaL=thetaL,
                            thetaU=thetaU,
                            nugget=None,
                            noise_estim=True,
                            optimizer='BFGS',
                            verbose=True,
                            wait_iter=3,
                            random_start=30,
                            likelihood='concentrated',
                            eval_budget=1e3)
    
    model.fit(X, y)
    
    def grad(model):
        f = EI(model)
        def __(x):
            _, dx = f(x, dx=True)
            return dx
        return __
    f = EI(model)
    dx = grad(model)
    
    fig0, ax0 = plt.subplots(1, 1, sharey=True, sharex=False,
                                figsize=(fig_width, fig_height),
                                subplot_kw={'aspect': 'auto'}, dpi=100)

    plot_contour_gradient(ax0, f, dx, x_lb, x_ub, title='Function',
                          n_level=15, n_per_axis=120)

    plt.tight_layout()
    plt.show()
