# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:44:21 2017

@author: Hao Wang
@email: wangronin@gmail.com
"""
import warnings
import numpy as np
from numpy import sqrt, exp, pi
from scipy.stats import norm
from abc import ABC, abstractmethod

# TODO: implement noisy handling infill criteria, e.g., EQI (expected quantile improvement)
# TODO: perphaps also enable acquisition function engineering here?
# meaning the combination of the acquisition functions
class InfillCriteria(ABC):
    def __init__(self, model=None, plugin=None, minimize=True):
        self.model = model
        self.minimize = minimize
        self.plugin = plugin
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        if model is not None:
            self._model = model
            assert hasattr(self._model, 'predict')

    @property
    def plugin(self):
        return self._plugin
    
    @plugin.setter
    def plugin(self, plugin):
        if plugin is None:
            if self._model is not None:
                self._plugin = np.min(self._model.y) if self.minimize \
                    else -1.0 * np.max(self._model.y)
        else:
            self._plugin = plugin if self.minimize else -1.0 * plugin

    @abstractmethod
    def __call__(self, X):
        raise NotImplementedError

    def _predict(self, X):
        y_hat, sd2 = self._model.predict(X, eval_MSE=True)
        sd = sqrt(sd2)
        if not self.minimize:
            y_hat = -y_hat
        return y_hat, sd

    def _gradient(self, X):
        y_dx, sd2_dx = self._model.gradient(X)
        if not self.minimize:
            y_dx = -y_dx
        return y_dx, sd2_dx

    def check_X(self, X):
        """Keep input as '2D' object 
        """
        return np.atleast_2d(X)
        # return [X] if not hasattr(X[0], '__iter__') else X

class UCB(InfillCriteria):
    def __init__(self, model, plugin=None, minimize=True, alpha=1e-10):
        """Upper Confidence Bound 
        """
        super(UCB, self).__init__(model, plugin, minimize)
        self.alpha = alpha

    def __call__(self, X, dx=False):
        X = self.check_X(X)
        y_hat, sd = self._predict(X)

        try:
            f_value = y_hat + self.alpha * sd
        except Exception: # in case of numerical errors
            f_value = 0

        if dx:
            y_dx, sd2_dx = self._gradient(X)
            sd_dx = sd2_dx / (2. * sd)
            try:
                f_dx = y_dx + self.alpha * sd_dx
            except Exception:
                f_dx = np.zeros((len(X[0]), 1))
            return f_value, f_dx 
        return f_value

class EI(InfillCriteria):
    # perhaps separate the gradient computation here
    def __call__(self, X, dx=False):
        X = self.check_X(X)
        n_sample = X.shape[0]
        y_hat, sd = self._predict(X)
        # if the Kriging variance is to small
        # TODO: check the rationale of 1e-6 and why the ratio if intended
        if hasattr(self._model, 'sigma2'):
            if sd / np.sqrt(self._model.sigma2) < 1e-6:
                return (0,  np.zeros((len(X[0]), 1))) if dx else 0.
        else: 
            # TODO: implement a counterpart of 'sigma2' for randomforest
            # or simply put a try...except around the code below
            if sd < 1e-10: 
                return (0,  np.zeros((len(X[0]), 1))) if dx else 0.
        try:
            xcr_ = self._plugin - y_hat
            xcr = xcr_ / sd
            xcr_prob, xcr_dens = norm.cdf(xcr), norm.pdf(xcr)
            f_value = xcr_ * xcr_prob + sd * xcr_dens
            if n_sample == 1:
                f_value = sum(f_value)
        except Exception: # in case of numerical errors
            # IMPORTANT: always keep the output in the same type
            f_value = 0

        if dx:
            y_dx, sd2_dx = self._gradient(X)
            sd_dx = sd2_dx / (2. * sd)
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

        coef = 1 - self.epsilon if y_hat > 0 else (1 + self.epsilon)
        try:
            xcr_ = self._plugin - coef * y_hat 
            xcr = xcr_ / sd
            f_value = norm.cdf(xcr)
        except Exception:
            f_value = 0.

        if dx:
            y_dx, sd2_dx = self._gradient(X)
            sd_dx = sd2_dx / (2. * sd)
            try:
                f_dx = -(coef * y_dx + xcr * sd_dx) * norm.pdf(xcr) / sd
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
    def __init__(self, t=1, *argv, **kwargs):
        """Moment-Generating Function of Improvement proposed in SMC'17 paper
        """
        super(MGFI, self).__init__(*argv, **kwargs)
        self.t = t

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        assert t > 0 and isinstance(t, float)
        self._t = t

    def __call__(self, X, dx=False):
        X = self.check_X(X)
        y_hat, sd = self._predict(X)
        n_sample = X.shape[0]

        # if the Kriging variance is to small
        # TODO: check the rationale of 1e-6 and why the ratio if intended
        if np.isclose(sd, 0):
            return (np.array([0.]), np.zeros((len(X[0]), 1))) if dx else 0.

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                y_hat_p = y_hat - self._t * sd ** 2.
                beta_p = (self._plugin - y_hat_p) / sd
                term = self._t * (self._plugin - y_hat - 1)
                f_ = norm.cdf(beta_p) * exp(term + self._t ** 2. * sd ** 2. / 2.)
                if n_sample == 1:
                    f_ = sum(f_)
            except Exception: # in case of numerical errors
                f_ = 0.

        if np.isinf(f_):
            f_ = 0.
            
        if dx:
            y_dx, sd2_dx = self._gradient(X)
            sd_dx = sd2_dx / (2. * sd)

            try:
                term = exp(self._t * (self._plugin + self._t * sd ** 2. / 2 - y_hat - 1))
                m_prime_dx = y_dx - 2. * self._t * sd * sd_dx
                beta_p_dx = -(m_prime_dx + beta_p * sd_dx) / sd
        
                f_dx = term * (norm.pdf(beta_p) * beta_p_dx + \
                    norm.cdf(beta_p) * ((self._t ** 2) * sd * sd_dx - self._t * y_dx))
            except Exception:
                f_dx = np.zeros((len(X[0]), 1))
            return f_, f_dx
        return f_
        
class GEI(InfillCriteria):
    def __init__(self, model, plugin=None, minimize=True, g=1):
        """Generalized Expected Improvement 
        """
        super(GEI, self).__init__(model, plugin, minimize)
        self.g = g

    def __call__(self, X, dx=False):
        pass

if __name__ == '__main__':

    # TODO: diagnostic plot for the gradient of Infill-Criteria
    # goes to unittest
    from GaussianProcess.trend import linear_trend, constant_trend
    from GaussianProcess import GaussianProcess
    from GaussianProcess.utils import plot_contour_gradient
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from deap import benchmarks

    np.random.seed(123)
    
    plt.ioff()
    fig_width = 16
    fig_height = 16
    
    noise_var = 0.
    def fitness(X):
        X = np.atleast_2d(X)
        return np.array([benchmarks.schwefel(x)[0] for x in X]) + \
            np.sqrt(noise_var) * np.random.randn(X.shape[0])
        
    dim = 2
    n_init_sample = 10

    x_lb = np.array([-5] * dim)
    x_ub = np.array([5] * dim)

    X = np.random.rand(n_init_sample, dim) * (x_ub - x_lb) + x_lb
    y = fitness(X)

    thetaL = 1e-5 * (x_ub - x_lb) * np.ones(dim)
    thetaU = 10 * (x_ub - x_lb) * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

    mean = linear_trend(dim, beta=None)
    model = GaussianProcess(mean=mean, corr='matern', theta0=theta0, thetaL=thetaL, thetaU=thetaU,
                            nugget=None, noise_estim=True, optimizer='BFGS', verbose=True,
                            wait_iter=3, random_start=10, eval_budget=50)
    
    model.fit(X, y)
    
    def grad(model):
        f = MGFI(model, t=10)
        def __(x):
            _, dx = f(x, dx=True)
            return dx
        return __
    
    t = 1
    infill = MGFI(model, t=t)
    infill_dx = grad(model)
    
    m = lambda x: model.predict(x)
    sd2 = lambda x: model.predict(x, eval_MSE=True)[1]

    m_dx = lambda x: model.gradient(x)[0]
    sd2_dx = lambda x: model.gradient(x)[1]
    
    if 1 < 2:
        fig0, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=False, sharex=False,
                                  figsize=(fig_width, fig_height),
                                  subplot_kw={'aspect': 'equal'}, dpi=100)
                                  
        gs1 = gridspec.GridSpec(1, 3)
        gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
    
        plot_contour_gradient(ax0, fitness, None, x_lb, x_ub, title='Noisy function',
                              n_level=20, n_per_axis=200)
        
        plot_contour_gradient(ax1, m, m_dx, x_lb, x_ub, title='GPR estimation',
                              n_level=20, n_per_axis=200)
                              
        plot_contour_gradient(ax2, sd2, sd2_dx, x_lb, x_ub, title='GPR variance',
                              n_level=20, n_per_axis=200)
        plt.tight_layout()
    
    fig1, ax3 = plt.subplots(1, 1, figsize=(fig_width, fig_height),
                             subplot_kw={'aspect': 'equal'}, dpi=100)
                             
    plot_contour_gradient(ax3, infill, infill_dx, x_lb, x_ub, title='Infill-Criterion',
                          is_log=True, n_level=50, n_per_axis=250)

    plt.tight_layout()
    plt.show()
