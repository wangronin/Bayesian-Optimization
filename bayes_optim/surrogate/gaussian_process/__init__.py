from __future__ import annotations

import warnings
from functools import partial
from typing import List, Tuple, Union

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor as sk_gpr
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

from ...search_space import RealSpace, SearchSpace
from ...utils import safe_divide
from .utils import MSLL, SMSE

# from sklearn.model_selection import cross_val_score


warnings.filterwarnings("ignore")

__author__ = "Hao Wang"
__all__ = ["SMSE", "MSLL", "GaussianProcess"]


class GaussianProcess:
    """Wrapper for sklearn's GPR model"""

    def __init__(
        self,
        domain: SearchSpace,
        codomain: SearchSpace = None,
        n_obj: int = 1,
        optimizer: str = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 0,
        normalize_y: bool = True,
        length_scale_bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = (1e-10, 1e10),
    ):
        assert isinstance(domain, RealSpace)
        assert codomain is None or isinstance(codomain, RealSpace)
        self.domain = domain
        # TODO: this is not used for now, which should implement restriction on the co-domain
        self.codomain = codomain
        self.dim = self.domain.dim
        self.n_obj = n_obj
        self.is_fitted = False

        self._set_length_scale_bounds(length_scale_bounds, np.atleast_2d(self.domain.bounds))
        self._set_kernels()

        if n_restarts_optimizer == 0:
            n_restarts_optimizer = int(3 * self.dim)

        self._gpr_cls = partial(
            sk_gpr,
            normalize_y=normalize_y,
            alpha=1e-8,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
        )

    def _set_length_scale_bounds(self, length_scale_bounds, bounds) -> np.ndarray:
        length_scale_bounds = np.atleast_2d(length_scale_bounds)
        if len(length_scale_bounds) == 1:
            length_scale_bounds = np.repeat(length_scale_bounds, self.dim, axis=0)
        self.length_scale_bounds = length_scale_bounds * (bounds[:, [1]] - bounds[:, [0]])

    def _set_kernels(self):
        kernel = [
            Matern(
                length_scale=np.ones(self.dim),
                length_scale_bounds=self.length_scale_bounds,
                nu=1.5,
            ),
            Matern(
                length_scale=np.ones(self.dim),
                length_scale_bounds=self.length_scale_bounds,
                nu=2.5,
            ),
            RBF(
                length_scale=np.ones(self.dim),
                length_scale_bounds=self.length_scale_bounds,
            ),
        ]
        self._kernel = [
            1.0 * k + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-15, 1e15)) for k in kernel
        ]

    def _check_dims(self, X: np.ndarray, y: np.ndarray):
        """check if the input/output dimensions are consistent with dimensions of the domain/co-domain"""
        if self.dim != X.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} variables, which does not match the dimension of the domain {self.dim}"
            )
        if len(y.shape) == 1:
            if self.n_obj != 1:
                raise ValueError(
                    f"y has one variable, which does not match "
                    f"the dimension of the co-domain {self.n_obj}"
                )
        else:
            if self.n_obj != y.shape[1]:
                raise ValueError(
                    f"y has {y.shape[1]} variables, which does not match "
                    f"the dimension of the co-domain {self.n_obj}"
                )

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianProcess:
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            `GaussianProcess` class instance.
        """
        X = np.array(X.tolist())
        self._check_dims(X, y)
        self._gprs = list()
        # cv = min(len(X), 3)

        for i in range(self.n_obj):
            # TODO: to verify this cross-validation procedure
            # score = -np.inf
            # kernel = None
            # for k in self._kernel:
            #     scores = cross_val_score(self._gpr_cls(kernel=k), X, y[:, i], cv=cv, scoring="r2", n_jobs=cv)
            #     score_ = np.nanmean(scores)
            #     if score_ > score:
            #         score = score_
            #         kernel = k

            gpr = self._gpr_cls(kernel=self._kernel[0]).fit(X, y[:, i])
            setattr(gpr, "_K_inv", None)
            self._gprs.append(gpr)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, eval_MSE: bool = False) -> np.ndarray:
        n_samples = X.shape[0]
        y_hat = np.zeros((n_samples, self.n_obj))
        if eval_MSE:
            std_hat = np.zeros((n_samples, self.n_obj))

        for i, gp in enumerate(self._gprs):
            out = gp.predict(X, return_std=eval_MSE)
            if eval_MSE:
                y_hat[:, i], std_hat[:, i] = out[0], out[1] ** 2
            else:
                y_hat[:, i] = out

        return (y_hat, std_hat) if eval_MSE else y_hat

    def gradient(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """compute the gradient of GPR's mean and variance w.r.t. the input points

        Parameters
        ----------
        X : np.ndarray
            the point at which gradients should be computed

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (grad of mean, grad of variance)
        """
        n_samples = X.shape[0]
        mean_dx = np.zeros((self.n_obj, n_samples, self.dim))
        var_dx = np.zeros((self.n_obj, n_samples, self.dim))

        for i, gp in enumerate(self._gprs):
            scale = getattr(gp, "_y_train_std")
            k = np.expand_dims(gp.kernel_(X, gp.X_train_), 2)  # shape (n_samples, N_train, 1)
            k_dx = self.kernel_dx(gp, X)  # shape (n_samples, dim, N_train)

            if getattr(gp, "_K_inv") is None:
                L_inv = solve_triangular(gp.L_.T, np.eye(gp.L_.shape[0]))
                setattr(gp, "_K_inv", L_inv.dot(L_inv.T))

            mean_dx[i, ...] = scale * k_dx @ gp.alpha_
            var_dx[i, ...] = -2.0 * scale ** 2 * np.squeeze(k_dx @ getattr(gp, "_K_inv") @ k)
        return mean_dx, var_dx

    def kernel_dx(self, gp: sk_gpr, X: np.ndarray) -> np.ndarray:
        """compute the gradient of the kernel function w.r.t. the input points

        Parameters
        ----------
        gp : sk_gpr
            a fitted `GaussianProcessRegressor` instance
        X : np.ndarray
            the location at which the gradient will be computed

        Returns
        -------
        np.ndarray
            the gradient of shape (n_samples, dim, n_train)
        """
        sigma2 = np.exp(gp.kernel_.k1.k1.theta)
        length_scale = np.exp(gp.kernel_.k1.k2.theta)
        kernel = gp.kernel_.k1.k2

        d = np.expand_dims(
            cdist(X / length_scale, gp.X_train_ / length_scale), 1
        )  # shape (n_samples, 1, n_train)
        X_ = np.expand_dims(X, 2)  # shape (n_samples, dim, 1)
        X_train_ = np.expand_dims(gp.X_train_.T, 0)  # shape (1, dim, n_train)
        # shape (n_samples, dim, n_train)
        dd = safe_divide(X_ - X_train_, d * np.expand_dims(length_scale ** 2, 1))

        if isinstance(kernel, Matern):
            nu = kernel.nu
            if nu == 0.5:
                g = -sigma2 * np.exp(-d) * dd
            elif nu == 1.5:
                g = -3 * sigma2 * np.exp(-np.sqrt(3) * d) * d * dd
            elif nu == 2.5:
                g = -5.0 / 3 * sigma2 * np.exp(-np.sqrt(5) * d) * (1 + np.sqrt(5) * d) * d * dd
        elif isinstance(kernel, RBF):
            g = -sigma2 * np.exp(-0.5 * d ** 2) * dd
        return g
