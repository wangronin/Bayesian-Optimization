#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:48:47 2017

@author: Hao Wang
@email: wangronin@gmail.com
"""

from abc import abstractmethod
import numpy as np
from numpy import newaxis, zeros, tile, eye, r_, c_, ones, array, atleast_2d

from sklearn.ensemble import RandomForestRegressor

class Trend(object):
    pass

class BasisExpansionTrend(Trend):
    def __init__(self, n_feature):
        """
        n_dim : the dimension of the function space of the trend function
        """
        self.n_feature = n_feature
        self.n_dim = None
        self.beta = None

    def set_beta(self, beta):
        if beta is not None:
            if not hasattr(beta, '__iter__'):
                beta = array([beta] * self.n_dim)
            beta = atleast_2d(beta).reshape(-1, 1)
            if len(beta) != self.n_dim:
                raise Exception('Shapes of beta and F do not match.')
        self.beta = beta

    def __str__(self):
        return self.__class__

    def __call__(self, X):
        if self.beta is None:
            raise Exception('beta is not set!')
        return self.F(X).dot(self.beta)
    
    @abstractmethod
    def F(self, X):
        "Evaluate the function basis as X"
    
    @abstractmethod
    def Jacobian(self, X):
        "Compute the Jacobian matrix of function basis"

    def check_input(self, X):
        # Check input shapes
        X = np.atleast_2d(X)
        if X.shape[1] != self.n_feature:
            X = X.T
        if X.shape[1] != self.n_feature:
            raise Exception('x does not have the right size!')
        return X

    def __eq__(self, trend_b):
        pass

    def __add__(self, trend_b):
        pass

# TODO: change all Jacobian function to numerator layout
class constant_trend(BasisExpansionTrend):
    """
    Zero order polynomial (constant, p = 1) regression model.

    x --> f(x) = 1

    """
    def __init__(self, n_feature, beta=None):
        super(constant_trend, self).__init__(n_feature)
        self.n_dim = 1
        self.set_beta(beta)

    def F(self, X):
        X = self.check_input(X)
        n_eval = X.shape[0]
        return ones((n_eval, 1))

    def Jacobian(self, x):
        self.check_input(x)
        return zeros((1, self.n_feature)) # numerator layout

class linear_trend(BasisExpansionTrend):
    """
    First order polynomial (linear, p = n+1) regression model.

    x --> f(x) = [ 1, x_1, ..., x_n ].T
    """
    def __init__(self, n_feature, beta=None):
        super(linear_trend, self).__init__(n_feature)
        self.n_dim = n_feature + 1
        self.set_beta(beta)
    
    # TODO: change this function name to __call__
    def F(self, X):
        X = self.check_input(X)
        n_eval = X.shape[0]
        return c_[ones(n_eval), X]

    def Jacobian(self, x):
        x = self.check_input(x)
        assert x.shape[0] == 1
        return r_[zeros((1, self.n_feature)), eye(self.n_feature)]

class quadratic_trend(BasisExpansionTrend):
    """
    Second order polynomial (quadratic, p = n*(n-1)/2+n+1) regression model.

    x --> f(x) = [ 1, { x_i, i = 1,...,n }, { x_i * x_j,  (i,j) = 1,...,n } ].T
                                                          i > j
    """
    def __init__(self, n_feature, beta=None):
        super(quadratic_trend, self).__init__(n_feature)
        self.set_beta(beta)
        self.n_dim = (n_feature + 1) * (n_feature + 2) / 2

    def F(self, X):
        X = self.check_input(X)
        n_eval = X.shape[0]
        f = c_[ones(n_eval), X]
        for k in range(self.n_feature):
            f = c_[f, X[:, k, np.newaxis] * X[:, k:]]
        return f

    def Jacobian(self, X):
        raise NotImplementedError

class NonparametricTrend(Trend):
    def __init__(self, X, y):
        self.regr = RandomForestRegressor(20)
        self.regr.fit(X, y)

    def __call__(self, X):
        return self.regr.predict(X)

if __name__ == '__main__':
    T = linear_trend(2, beta=(1, 2, 10))

    X = np.random.randn(5, 2)
    print(T(X))
    print(T.Jacobian(X))


# TODO: remove those functions
# legacy functions
def constant(x):

    """
    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    """
    x = np.asarray(x, dtype=np.float64)
    n_eval = x.shape[0]
    f = np.ones([n_eval, 1])
    return f


def linear(x):
    """
    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    """
    x = np.asarray(x, dtype=np.float64)
    n_eval = x.shape[0]
    f = np.hstack([np.ones([n_eval, 1]), x])
    return f


def quadratic(x):
    """
    Second order polynomial (quadratic, p = n*(n-1)/2+n+1) regression model.

    x --> f(x) = [ 1, { x_i, i = 1,...,n }, { x_i * x_j,  (i,j) = 1,...,n } ].T
                                                          i > j

    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    """

    x = np.asarray(x, dtype=np.float64)
    n_eval, n_features = x.shape
    f = np.hstack([np.ones([n_eval, 1]), x])
    for k in range(n_features):
        f = np.hstack([f, x[:, k, np.newaxis] * x[:, k:]])

    return f
