from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor as sk_gpr
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern

from ...search_space import RealSpace, SearchSpace
from ...utils import safe_divide
from .utils import MSLL, SMSE

__author__ = "Hao Wang"
__all__ = ["SMSE", "MSLL", "GaussianProcess"]


class GaussianProcess:
    """Wrapper for sklearn's GPR model"""

    def __init__(
        self,
        domain: SearchSpace,
        codomain: SearchSpace = None,
        n_obj: int = 1,
        kernel: Optional[Kernel] = None,
        alpha: float = 1e-4,
        optimizer: str = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 0,
        normalize_y: bool = True,
        length_scale_bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = (1e-5, 1e5),
        constant_value_bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = (1e-5, 1e5),
        **kwargs,
    ):
        assert isinstance(domain, RealSpace)
        assert codomain is None or isinstance(codomain, RealSpace)
        self.domain = domain
        # TODO: this is not used for now, which should implement restriction on the co-domain
        self.codomain = codomain
        self.dim = self.domain.dim
        self.n_obj = n_obj
        self.nu = 1.5
        self.is_fitted = False

        if kernel is None:
            bounds = np.atleast_2d(self.domain.bounds)
            length_scale_bounds = self._set_length_scale_bounds(length_scale_bounds, bounds)
            kernel = Matern(
                length_scale=np.ones(self.dim),
                length_scale_bounds=length_scale_bounds,
                nu=self.nu,
            )
            kernel *= ConstantKernel(constant_value=1.0, constant_value_bounds=constant_value_bounds)

        if n_restarts_optimizer == 0:
            n_restarts_optimizer = int(5 * self.dim)

        self._gpr = list()
        # create GPRs for each objective function
        for _ in range(n_obj):
            self._gpr.append(
                sk_gpr(
                    kernel=kernel,
                    normalize_y=normalize_y,
                    alpha=alpha,
                    optimizer=optimizer,
                    n_restarts_optimizer=n_restarts_optimizer,
                    **kwargs,
                )
            )

    def _set_length_scale_bounds(self, length_scale_bounds, bounds) -> np.ndarray:
        length_scale_bounds = np.atleast_2d(length_scale_bounds)
        if len(length_scale_bounds) == 1:
            length_scale_bounds = np.repeat(length_scale_bounds, self.dim, axis=0)
        length_scale_bounds *= bounds[:, [1]] - bounds[:, [0]]
        return length_scale_bounds

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
        self._check_dims(X, y)
        for i, gp in enumerate(self._gpr):
            gp.fit(X, y[:, i])
            setattr(gp, "_K_inv", None)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, eval_MSE: bool = False) -> np.ndarray:
        n_samples = X.shape[0]
        y_hat = np.zeros((n_samples, self.n_obj))
        if eval_MSE:
            std_hat = np.zeros((n_samples, self.n_obj))

        for i, gp in enumerate(self._gpr):
            out = gp.predict(X, return_std=eval_MSE)
            if eval_MSE:
                y_hat[:, i], std_hat[:, i] = out[0], out[1] ** 2
            else:
                y_hat[:, i] = out

        return (y_hat, std_hat) if eval_MSE else y_hat

    def gradient(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        mean_dx = np.zeros((self.n_obj, n_samples, self.dim))
        var_dx = np.zeros((self.n_obj, n_samples, self.dim))

        for i, gp in enumerate(self._gpr):
            scale = getattr(gp, "_y_train_std")
            k = np.expand_dims(gp.kernel_(X, gp.X_train_), 2)  # shape (n_samples, N_train, 1)
            k_dx = self._kernel_dx(gp, X)  # shape (n_samples, dim, N_train)

            if getattr(gp, "_K_inv") is None:
                L_inv = solve_triangular(gp.L_.T, np.eye(gp.L_.shape[0]))
                setattr(gp, "_K_inv", L_inv.dot(L_inv.T))

            mean_dx[i, ...] = scale * k_dx @ gp.alpha_
            var_dx[i, ...] = -2.0 * scale ** 2 * np.squeeze(k_dx @ getattr(gp, "_K_inv") @ k)
        return mean_dx, var_dx

    def _kernel_dx(self, gp: sk_gpr, X: np.ndarray) -> np.ndarray:
        sigma2 = np.exp(gp.kernel_.theta[0])
        length_scale = np.exp(gp.kernel_.theta[1:])

        d = np.expand_dims(
            cdist(X / length_scale, gp.X_train_ / length_scale), 1
        )  # shape (n_samples, 1, n_train)
        X_ = np.expand_dims(X, 2)  # shape (n_samples, dim, 1)
        X_train_ = np.expand_dims(gp.X_train_.T, 0)  # shape (1, dim, n_train)
        # shape (n_samples, dim, n_train)
        dd = safe_divide(X_ - X_train_, d * np.expand_dims(length_scale ** 2, 1))

        if self.nu == 0.5:
            g = -sigma2 * np.exp(-d) * dd
        elif self.nu == 1.5:
            g = -3 * sigma2 * np.exp(-np.sqrt(3) * d) * d * dd
        elif self.nu == 2.5:
            g = -5.0 / 3 * sigma2 * np.exp(-np.sqrt(5) * d) * (1 + np.sqrt(5) * d) * d * dd
        return g
