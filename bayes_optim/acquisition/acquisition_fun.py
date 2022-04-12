import warnings
from abc import ABC, abstractmethod
from multiprocessing.sharedctypes import Value
from typing import Tuple, Union

import numpy as np
from bayes_optim.mylogging import eprintf
from numpy import exp, sqrt
from scipy.stats import norm

from ..surrogate import GaussianProcess, RandomForest

__authors__ = ["Hao Wang"]

# TODO: implement noisy handling infill criteria, e.g., EQI (expected quantile improvement)
# TODO: perphaps also enable acquisition function engineering here?
# TODO: multi-objective extension
# TODO: maybe handle the minimization/maximization in a abstract problem class
# TODO: maybe attach the optimizer and constraints to the `AcquisitionFunction`
# TODO: add an interface to get the free parameters of each acqusition function


class AcquisitionFunction(ABC):
    """Base class for acquisition functions"""

    def __init__(self, model: Union[GaussianProcess, RandomForest], minimize: bool = True):
        """Base class for acquisition functions

        Args:
            model (Union[GaussianProcess, RandomForest], optional): the surrogate model on which the
                acquisition function is defined.
            minimize (bool, optional): whether the underlying problem is subject to minimization.
                Defaults to True.
        """
        self.model = model
        self.minimize = minimize

    @property
    def model(self) -> Union[GaussianProcess, RandomForest]:
        return self._model

    @model.setter
    def model(self, model: Union[GaussianProcess, RandomForest]):
        if model is None:
            raise ValueError("model cannot be None")
        self._model = model
        assert hasattr(self._model, "predict")

    @abstractmethod
    def __call__(self, X: np.ndarray):
        raise NotImplementedError

    def _predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """wrap around `model.predict`

        Args:
            X (np.ndarray): the input points to be predicted

        Returns:
            Tuple[np.ndarray, np.ndarray]: the predicted value and its standard deviation
        """
        y_hat, sd2 = self._model.predict(X, eval_MSE=True)
        if not self.minimize:
            y_hat = -1 * y_hat
        return y_hat, sqrt(sd2)

    def _gradient(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """wrap around `model.gradient`: reverse the gradient direction for maximization problem

        Args:
            X (np.ndarray): the input points to be predicted

        Returns:
            Tuple[np.ndarray, np.ndarray]: the gradient of prediction and its standard deviation w.r.t.
                the decision variables
        """
        X_ = np.array(X, dtype=float)
        y_dx, sd2_dx = self._model.gradient(X_)
        if not self.minimize:
            y_dx = -1.0 * y_dx
        return y_dx.T, sd2_dx.T

    def check_X(self, X: np.ndarray) -> np.ndarray:
        """Check the shape of the input, which is enforced to a 2D array"""
        return np.atleast_2d(X)


class ImprovementBased(AcquisitionFunction):
    def __init__(self, plugin: float = None, **kwargs):
        super().__init__(**kwargs)
        self.plugin = plugin

    @property
    def plugin(self) -> float:
        return self._plugin

    @plugin.setter
    def plugin(self, plugin: float):
        if plugin is None:
            if hasattr(self._model, "y"):
                self._plugin = np.min(self._model.y) if self.minimize else -1.0 * np.max(self._model.y)
            else:
                self._plugin = None
        else:
            self._plugin = plugin if self.minimize else -1.0 * plugin


class UCB(AcquisitionFunction):
    def __init__(self, alpha: float = 0.5, **kwargs):
        """Upper Confidence Bound

        \alpha(x) = m(x) + \alpha * \sigma(x),

        where m(x) the predicted value and \sigma(x) is the uncertainty of the prediction
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float):
        assert alpha > 0
        self._alpha = alpha

    def __call__(self, X: np.ndarray, return_dx: bool = False):
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
    """Expected Improvement"""

    def __call__(
        self, X: np.ndarray, return_dx: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        X = self.check_X(X)
        n_sample = X.shape[0]
        xi = 0.0

        if return_dx:
            mu, std, mu_grad, std_grad = self._model.predict(
                X, return_std=True, return_mean_grad=True, return_std_grad=True
            )
        else:
            mu, std = self._model.predict(X, return_std=True)
        # if the Kriging variance is to small
        # TODO: check the rationale of 1e-6 and why the ratio if intended
        if hasattr(self._model, "sigma2"):
            if std / np.sqrt(self._model.sigma2) < 1e-6:
                return (0, np.zeros((len(X[0]), 1))) if return_dx else 0
        else:
            # TODO: implement a counterpart of 'sigma2' for randomforest
            # or simply put a try...except around the code below
            if std < 1e-10:
                return (0, np.zeros((len(X[0]), 1))) if return_dx else 0
        try:
            impr = self.plugin - xi - mu
            xcr = impr / std
            cdf, pdf = norm.cdf(xcr), norm.pdf(xcr)
            value = impr * cdf + std * pdf
            if n_sample == 1:
                value = sum(value)
        except Exception:  # in case of numerical errors
            # IMPORTANT: always keep the output in the same type
            value = 0

        if return_dx:
            try:
                improve_grad = -mu_grad * std - std_grad * impr
                improve_grad /= std**2
                cdf_grad = improve_grad * pdf
                pdf_grad = -impr * cdf_grad
                exploit_grad = -mu_grad * cdf - pdf_grad
                explore_grad = std_grad * pdf + pdf_grad
                dx = exploit_grad + explore_grad
            except Exception:
                dx = np.zeros((len(X[0]), 1))
            return value, dx
        return value


class EpsilonPI(ImprovementBased):
    """epsilon-Probability of Improvement"""

    def __init__(self, epsilon=1e-10, **kwargs):
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
    """Moment Generating Function of the Improvement (MGFI)

    M(x; t) = \Phi((plugin - m(x) + s(x)^2 * t - 1) / s(x)) * exp((plugin - m(x) - 1) * t + (s(x) * t)^2 / 2)

    References:

        Wang, Hao, Bas van Stein, Michael Emmerich, and Thomas Back. "A new acquisition function for Bayesian
        optimization based on the moment-generating function." In 2017 IEEE International Conference on
        Systems, Man, and Cybernetics (SMC), pp. 507-512. IEEE, 2017.
    """

    def __init__(self, t: float = 1, **kwargs):
        """Moment-Generating Function of Improvement proposed in SMC'17 paper"""
        super().__init__(**kwargs)
        self.t = t

    @property
    def t(self) -> float:
        return self._t

    @t.setter
    def t(self, t: float):
        assert t > 0
        t = min(t, 22.36)  # NOTE: huge `t` values would cause numerical overflow
        self._t = t

    def __call__(
        self, X: np.ndarray, return_dx: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
                y_hat_p = y_hat - self._t * sd**2.0
                beta_p = (self._plugin - y_hat_p) / sd
                term = self._t * (self._plugin - y_hat - 1)
                f_ = norm.cdf(beta_p) * exp(term + self._t**2.0 * sd**2.0 / 2.0)
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
                    term = exp(self._t * (self._plugin + self._t * sd**2.0 / 2 - y_hat - 1))
                    m_prime_dx = y_dx - 2.0 * self._t * sd * sd_dx
                    beta_p_dx = -(m_prime_dx + beta_p * sd_dx) / sd

                    f_dx = term * (
                        norm.pdf(beta_p) * beta_p_dx
                        + norm.cdf(beta_p) * ((self._t**2) * sd * sd_dx - self._t * y_dx)
                    )
                except Exception:
                    f_dx = np.zeros((len(X[0]), 1))
            return f_, f_dx
        return f_


# class GEI(ImprovementBased):
#     def __init__(self, g=1, **kwargs):
#         """Generalized Expected Improvement"""
#         super().__init__(**kwargs)
#         self.g = g

#     @property
#     def g(self):
#         return self._g

#     @g.setter
#     def g(self, g):
#         g = int(g)
#         assert g >= 0
#         self._g = g

#     def __call__(self, X, return_dx=False):
#         # TODO: implement this!!!
#         raise NotImplementedError
