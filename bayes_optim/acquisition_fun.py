import warnings
from abc import ABC, abstractmethod

import numpy as np
from numpy import exp, sqrt
from scipy.stats import norm

__authors__ = ["Hao Wang"]

# TODO: add annotations
# TODO: implement noisy handling infill criteria, e.g., EQI (expected quantile improvement)
# TODO: perphaps also enable acquisition function engineering here?
# TODO: multi-objective extension


class AcquisitionFunction(ABC):
    def __init__(self, model=None, minimize=True):
        self.model = model
        self.minimize = minimize

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if model is not None:
            self._model = model
            assert hasattr(self._model, "predict")
        else:
            self._model = None

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
        X_ = np.array(X, dtype=float)
        y_dx, sd2_dx = self._model.gradient(X_)
        if not self.minimize:
            y_dx = -y_dx
        return y_dx, sd2_dx

    def check_X(self, X):
        """Keep input as '2D' object"""
        return np.atleast_2d(X)


class ImprovementBased(AcquisitionFunction):
    def __init__(self, plugin=None, **kwargs):
        super().__init__(**kwargs)
        self.plugin = plugin

    @property
    def plugin(self):
        return self._plugin

    @plugin.setter
    def plugin(self, plugin):
        if plugin is None:
            if self._model is not None:
                self._plugin = (
                    np.min(self._model.y) if self.minimize else -1.0 * np.max(self._model.y)
                )
            else:
                self._plugin = None
        else:
            self._plugin = plugin if self.minimize else -1.0 * plugin


class UCB(AcquisitionFunction):
    def __init__(self, alpha=0.5, **kwargs):
        """Upper Confidence Bound"""
        super().__init__(**kwargs)
        self.alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        assert alpha > 0
        self._alpha = alpha

    def __call__(self, X, return_dx=False):
        X = self.check_X(X)
        n_sample = X.shape[0]
        y_hat, sd = self._predict(X)

        try:
            f_value = y_hat + self.alpha * sd
            if n_sample == 1:
                f_value = sum(f_value)
        except Exception:  # in case of numerical errors
            f_value = 0

        if return_dx:
            y_dx, sd2_dx = self._gradient(X)
            sd_dx = sd2_dx / (2.0 * sd)
            try:
                f_dx = y_dx + self.alpha * sd_dx
            except Exception:
                f_dx = np.zeros((len(X[0]), 1))
            return f_value, f_dx
        return f_value


class EI(ImprovementBased):
    def __call__(self, X, return_dx=False):
        X = self.check_X(X)
        n_sample = X.shape[0]
        y_hat, sd = self._predict(X)

        # if the Kriging variance is to small
        # TODO: check the rationale of 1e-6 and why the ratio if intended
        if hasattr(self._model, "sigma2"):
            if sd / np.sqrt(self._model.sigma2) < 1e-6:
                return (0, np.zeros((len(X[0]), 1))) if return_dx else 0
        else:
            # TODO: implement a counterpart of 'sigma2' for randomforest
            # or simply put a try...except around the code below
            if sd < 1e-10:
                return (0, np.zeros((len(X[0]), 1))) if return_dx else 0
        try:
            xcr_ = self._plugin - y_hat
            xcr = xcr_ / sd
            xcr_prob, xcr_dens = norm.cdf(xcr), norm.pdf(xcr)
            f_value = xcr_ * xcr_prob + sd * xcr_dens
            if n_sample == 1:
                f_value = sum(f_value)
        except Exception:  # in case of numerical errors
            # IMPORTANT: always keep the output in the same type
            f_value = 0

        if return_dx:
            y_dx, sd2_dx = self._gradient(X)
            sd_dx = sd2_dx / (2.0 * sd)
            try:
                f_dx = -y_dx * xcr_prob + sd_dx * xcr_dens
            except Exception:
                f_dx = np.zeros((len(X[0]), 1))
            return f_value, f_dx
        return f_value


class EpsilonPI(ImprovementBased):
    def __init__(self, epsilon=1e-10, **kwargs):
        """epsilon-Probability of Improvement
        # TODO: verify the implementation
        """
        super(EpsilonPI, self).__init__(**kwargs)
        self.epsilon = epsilon

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, eps):
        assert eps > 0
        self._epsilon = eps

    def __call__(self, X, return_dx=False):
        X = self.check_X(X)
        y_hat, sd = self._predict(X)

        coef = 1 - self._epsilon if y_hat > 0 else (1 + self._epsilon)
        try:
            xcr_ = self._plugin - coef * y_hat
            xcr = xcr_ / sd
            f_value = norm.cdf(xcr)
        except Exception:
            f_value = 0.0

        if return_dx:
            y_dx, sd2_dx = self._gradient(X)
            sd_dx = sd2_dx / (2.0 * sd)
            try:
                f_dx = -(coef * y_dx + xcr * sd_dx) * norm.pdf(xcr) / sd
            except Exception:
                f_dx = np.zeros((len(X[0]), 1))
            return f_value, f_dx
        return f_value


class PI(EpsilonPI):
    def __init__(self, **kwargs):
        """Probability of Improvement"""
        kwargs.update({"epsilon": 0})
        super().__init__(**kwargs)


class MGFI(ImprovementBased):
    def __init__(self, t=1, **kwargs):
        """Moment-Generating Function of Improvement proposed in SMC'17 paper"""
        super().__init__(**kwargs)
        self.t = t

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        assert t > 0
        t = min(t, 22.36)  # bigger `t` value would cause numerical overflow
        self._t = t

    def __call__(self, X, return_dx=False):
        X = self.check_X(X)
        y_hat, sd = self._predict(X)
        n_sample = X.shape[0]

        # if the Kriging variance is to small
        # TODO: check the rationale of 1e-6 and why the ratio if intended
        if np.isclose(sd, 0):
            return (np.array([0.0]), np.zeros((len(X[0]), 1))) if return_dx else 0.0

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                y_hat_p = y_hat - self._t * sd ** 2.0
                beta_p = (self._plugin - y_hat_p) / sd
                term = self._t * (self._plugin - y_hat - 1)
                f_ = norm.cdf(beta_p) * exp(term + self._t ** 2.0 * sd ** 2.0 / 2.0)
                if n_sample == 1:
                    f_ = sum(f_)
            except Exception:  # in case of numerical errors
                f_ = 0.0

        if np.isinf(f_):
            f_ = 0.0

        if return_dx:
            y_dx, sd2_dx = self._gradient(X)
            sd_dx = sd2_dx / (2.0 * sd)

            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    term = exp(self._t * (self._plugin + self._t * sd ** 2.0 / 2 - y_hat - 1))
                    m_prime_dx = y_dx - 2.0 * self._t * sd * sd_dx
                    beta_p_dx = -(m_prime_dx + beta_p * sd_dx) / sd

                    f_dx = term * (
                        norm.pdf(beta_p) * beta_p_dx
                        + norm.cdf(beta_p) * ((self._t ** 2) * sd * sd_dx - self._t * y_dx)
                    )
                except Exception:
                    f_dx = np.zeros((len(X[0]), 1))
            return f_, f_dx
        return f_


class GEI(ImprovementBased):
    def __init__(self, g=1, **kwargs):
        """Generalized Expected Improvement"""
        super().__init__(**kwargs)
        self.g = g

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, g):
        g = int(g)
        assert g >= 0
        self._g = g

    def __call__(self, X, return_dx=False):
        # TODO: implement this!!!
        raise NotImplementedError
