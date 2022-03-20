from abc import ABC, abstractmethod

import numpy as np
from numpy import array, atleast_2d, c_, eye, newaxis, ones, r_, tile, zeros
from sklearn.ensemble import RandomForestRegressor

__author__ = "Hao Wang"


class BasisExpansionTrend:
    def __init__(self, n_feature, n_dim=None, beta=None):
        """n_dim : the dimension of the function space of the trend function"""
        self.n_feature = int(n_feature)
        self.n_dim = int(n_dim)
        self.beta = beta

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        if beta is not None:
            if not hasattr(beta, "__iter__"):
                beta = array([beta] * self.n_dim)
            beta = atleast_2d(beta).reshape(-1, 1)
            if len(beta) != self.n_dim:
                raise Exception("Shapes of beta and F do not match.")
        self._beta = beta

    def __str__(self):
        return self.__class__

    def __call__(self, X):
        if self._beta is None:
            raise Exception("beta is not set!")
        return self.F(X).dot(self._beta)

    @abstractmethod
    def F(self, X):
        "Evaluate the function basis as X"

    @abstractmethod
    def Jacobian(self, X):
        "Compute the Jacobian matrix of function basis"

    @abstractmethod
    def Hessian(self, X):
        "Compute the Hessian tensor of function basis"

    def check_input(self, X):
        # Check input shapes
        X = np.atleast_2d(X)
        if X.shape[1] != self.n_feature:
            X = X.T
        if X.shape[1] != self.n_feature:
            raise Exception("X does not have the right size!")
        return X

    def __eq__(self, trend_b):
        pass

    def __add__(self, trend_b):
        pass


# TODO: change all Jacobian function to denominator layout
# TODO: also to unify the naming of classes.
class constant_trend(BasisExpansionTrend):
    """Zero order polynomial (constant, p = 1) regression model.

    x --> f(x) = 1

    """

    def __init__(self, n_feature: int, beta: float = None):
        super(constant_trend, self).__init__(n_feature, 1, beta)

    def F(self, X):
        X = self.check_input(X)
        n_eval = X.shape[0]
        return ones((n_eval, 1))

    def Jacobian(self, x):
        self.check_input(x)
        return zeros((1, self.n_feature))  # numerator layout

    def Hessian(self, x):
        self.check_input(x)
        # TODO: unified the layout for matrix calculus here!!!
        return zeros((self.n_feature, self.n_feature, self.n_dim))  # denominator layout


class linear_trend(BasisExpansionTrend):
    """First order polynomial (linear, p = n+1) regression model.

    x --> f(x) = [ 1, x_1, ..., x_n ].T
    """

    def __init__(self, n_feature, beta=None):
        super(linear_trend, self).__init__(n_feature, n_feature + 1, beta)

    def F(self, X):
        X = self.check_input(X)
        n_eval = X.shape[0]
        return c_[ones(n_eval), X]

    def Jacobian(self, x):
        x = self.check_input(x)
        assert x.shape[0] == 1
        return r_[zeros((1, self.n_feature)), eye(self.n_feature)]

    def Hessian(self, x):
        self.check_input(x)
        # TODO: unified the layout for matrix calculus here!!!
        return zeros((self.n_feature, self.n_feature, self.n_dim))  # denominator layout


class quadratic_trend(BasisExpansionTrend):
    """
    Second order polynomial (quadratic, p = n * (n-1) / 2 + n + 1) regression model.

    x --> f(x) = [ 1, { x_i, i = 1,...,n }, { x_i * x_j,  (i,j) = 1,...,n } ].T
                                                          i > j
    """

    def __init__(self, n_feature, beta=None):
        super(quadratic_trend, self).__init__(n_feature, (n_feature + 1) * (n_feature + 2) / 2, beta)

    def F(self, X):
        X = self.check_input(X)
        n_eval = X.shape[0]
        f = c_[ones(n_eval), X]
        for k in range(self.n_feature):
            f = c_[f, X[:, k, np.newaxis] * X[:, k:]]
        return f

    def Jacobian(self, X):
        raise NotImplementedError

    def Hessian(self, X):
        raise NotImplementedError


class NonparametricTrend:
    def __init__(self, X, y):
        self.regr = RandomForestRegressor(20)
        self.regr.fit(X, y)

    def __call__(self, X):
        return self.regr.predict(X)
