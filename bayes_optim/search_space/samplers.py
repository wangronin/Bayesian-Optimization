from __future__ import annotations

from abc import ABC
from typing import Callable, Tuple

import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
# from scipy.optimize import root_scalar
from scipy.stats import norm
from torch import Tensor
from torch.nn import Module

from .._exception import ConstraintEvaluationError

__author__ = "Hao Wang"


class SCMC:
    r"""Sequential Constrained Monte Carlo sampling

    References

    .. [GolchiL15]
        Golchi, Shirin, and Jason L. Loeppky. "Monte Carlo based designs for constrained domains."
        arXiv preprint arXiv:1512.07328 (2015).

    TODO: consider adopting MIES-like self-adaptation for the MH algorithm
    """

    def __init__(
        self,
        sample_space,
        constraints: Callable,
        target_dist: Callable = lambda x: 1,
        metropolis_hastings_step: int = 15,
        tol: float = 1e-2,
    ):
        """Sequential Constrained Monte Carlo sampling

        Parameters
        ----------
        sample_space : SearchSpace
            the sample space
        constraints : Callable
            constraint functions, which could represent inequalities or equalities
        target_dist : Callable, optional
            the target probability density, by default uniform (lambda x: 1)
        metropolis_hastings_step : int, optional
            the number of MCMC iterations, by default 15
        tol : float, optional
            the tolerance on the constraint
        """
        if hasattr(constraints, "__call__"):
            constraints = [constraints]
        assert hasattr(target_dist, "__call__")

        self.constraints = constraints
        self.target_dist = target_dist
        self.sample_space = sample_space
        self.dim = sample_space.dim
        self.mh_steps = metropolis_hastings_step
        self.nu_target = tol / 8  # 8-sigma CI leads to ~1.24e-15 significance
        self.nu0 = 10
        self.nu_schedule = np.logspace(
            np.log10(self.nu0), np.log10(self.nu_target), base=10, num=15
        )

        # index of each type of variables in the dataframe
        self.id_r = self.sample_space.real_id  # index of continuous variable
        self.id_i = self.sample_space.integer_id  # index of integer variable
        self.id_d = self.sample_space.categorical_id  # index of categorical variable

        # the number of variables per each type
        self.N_r = len(self.id_r)
        self.N_i = len(self.id_i)
        self.N_d = len(self.id_d)

        self.bounds_r = np.asarray([self.sample_space.bounds[_] for _ in self.id_r])
        self.bounds_i = np.asarray([self.sample_space.bounds[_] for _ in self.id_i])
        self.bounds_d = [self.sample_space.bounds[_] for _ in self.id_d]

    def _log_posterior(self, x: np.ndarray, nu: float) -> float:
        r"""Log-posterior \log P(x) + \sum_{i=1}^K \log \Phi(-C_i(x) / \nu)

        Parameters
        ----------
        x : np.ndarray
            a point in the search space
        nu : float

        Returns
        -------
        float
            the log-probability
        """
        try:
            c = np.array([cons(x) for cons in self.constraints])
        except Exception as e:
            raise ConstraintEvaluationError(x, str(e)) from None
        return np.log(self.target_dist(x)) + np.sum(norm.logcdf(-1 * c / nu))

    def _rproposal(self, X: np.ndarray, t: int) -> np.ndarray:
        X_ = X.copy()
        self._rproposal_real(X_, t)
        self._rproposal_integer(X_, t)
        self._rproposal_discrete(X_, t)
        return X_

    def _rproposal_real(self, X: np.ndarray, t: int):
        if self.N_r > 0:
            X_ = X[:, self.id_r].astype(float)
            q = np.std(X_, axis=0) / (t + 1)
            X[:, self.id_r] = np.clip(
                X_ + q * np.random.randn(*X_.shape), self.bounds_r[:, 0], self.bounds_r[:, 1]
            )

    def _rproposal_integer(self, X: np.ndarray, t: int):
        if self.N_i > 0:
            X_ = X[:, self.id_i].astype(int)
            p = norm.cdf(np.std(X_, axis=0) / (t + 1))
            X[:, self.id_i] = np.clip(
                X_ + np.random.geometric(p, X_.shape) - np.random.geometric(p, X_.shape),
                self.bounds_i[:, 0],
                self.bounds_i[:, 1],
            )

    def _rproposal_discrete(self, X: np.ndarray, t: int):
        if self.N_d > 0:
            N = len(X)
            prob = max(0.5 * np.exp(-1 * t), 1 / self.N_d)
            for i in self.id_d:
                levels = np.array(self.sample_space.bounds[i])
                idx = np.random.rand(N) < prob
                X[idx, i] = levels[np.random.randint(0, len(levels), sum(idx))]

    def _ess(self, nu: float, nu_ref: float, X: np.ndarray) -> float:
        """effective sample size"""
        w = self.get_weights(nu, nu_ref, X)
        v = 0 if np.any(np.isnan(w)) else 1 / np.sum(w ** 2)
        return v - len(X) / 2

    def get_weights(self, nu, nu_ref, X):
        w = np.exp([self._log_posterior(x, nu) - self._log_posterior(x, nu_ref) for x in X])
        return w / np.sum(w)

    def _metropolis_hastings(self, X: np.ndarray, nu: float, t: int) -> np.ndarray:
        r"""Metropolis-Hastings algorithms to sample from a distribution given
        by `self._log_posterior`.

        Parameters
        ----------
        X : np.ndarray
            the initial sample
        nu : float
            parameter in `self._log_posterior`
        t : int
            iteration counter

        Returns
        -------
        np.ndarray
            (dependent) sample points from `self._log_posterior`
        """
        N = len(X)  # sample size
        lp = np.array([self._log_posterior(x, nu) for x in X])  # log-probability
        for _ in range(self.mh_steps):
            X_ = self._rproposal(X, t)
            for i in range(self.dim):
                X_dim = X.copy()
                X_dim[:, i] = X_[:, i]
                lp_ = np.array([self._log_posterior(x_, nu) for x_ in X_dim])
                prob = np.clip(np.exp(lp_ - lp), 0, 1)
                mask = np.random.rand(N) <= prob
                X[mask, i] = X_[mask, i]
                lp[mask] = lp_[mask]
        return X

    def sample(self, N: int) -> np.ndarray:
        """Draw a sample under constraints

        Parameters
        ----------
        N : int
            sample size

        Returns
        -------
        np.ndarray
            the sample
        """
        # draw the initial samples u.a.r. from `self.search_space`
        X = getattr(self.sample_space, "_sample")(N, method="LHS")
        for t, nu in enumerate(self.nu_schedule):
            nu_ = nu  # `nu_` -> the old `nu`
            # nu = self.nu_schedule[t]
            # if self._ess(self.nu_target, nu, X) > 0:
            #     nu *= 0.3
            # else:
            #     try:
            #         nu = (
            #             10
            #             ** root_scalar(
            #                 lambda log_nu, *args: self._ess(10 ** log_nu, *args),
            #                 args=(nu_, X),
            #                 bracket=(
            #                     np.log10(0.001 if t < 2 else self.nu_target),
            #                     np.log10(nu_),
            #                 ),
            #             ).root
            #         )
            #     except ValueError:
            #         return X
            w = self.get_weights(nu, nu_, X)
            if any(np.isnan(w)):
                w = np.ones(N) / N

            # resampling with replacement to increase the fraction of samples that are
            # more likely distributed from the new probability law, which is controlled by `nu`
            if N >= 10:
                idx = np.random.choice(N, N, p=w)
                X = X[idx, :]

            X = self._metropolis_hastings(X, nu, t)
        return X


class MCSampler(Module, ABC):
    r"""Abstract base class for Samplers.

    Attributes:
        resample: If `True`, re-draw samples in each `forward` evaluation -
            this results in stochastic acquisition functions (and thus should
            not be used with deterministic optimization algorithms).
        collapse_batch_dims: If True, collapse the t-batch dimensions of the
            produced samples to size 1. This is useful for preventing sampling
            variance across t-batches.
    """

    def __init__(self, num_samples: int, batch_range: Tuple[int, int] = (0, -2)) -> None:
        r"""Abstract base class for Samplers.

        Args:
            batch_range: The range of t-batch dimensions in the `base_sample_shape`
                used by `collapse_batch_dims`. The t-batch dims are
                batch_range[0]:batch_range[1]. By default, this is (0, -2),
                for the case where the non-batch dimensions are -2 (q) and
                -1 (d) and all dims in the front are t-batch dims.
        """
        super().__init__()
        self.batch_range = batch_range
        self.register_buffer("base_samples", None)
        self._sample_shape = torch.Size([num_samples])

    @property
    def batch_range(self) -> Tuple[int, int]:
        r"""The t-batch range."""
        return tuple(self._batch_range.tolist())

    @batch_range.setter
    def batch_range(self, batch_range: Tuple[int, int]):
        r"""Set the t-batch range and clear base samples.

        Args:
            batch_range: The range of t-batch dimensions in the `base_sample_shape`
                used by `collapse_batch_dims`. The t-batch dims are
                batch_range[0]:batch_range[1]. By default, this is (0, -2),
                for the case where the non-batch dimensions are -2 (q) and
                -1 (d) and all dims in the front are t-batch dims.
        """
        # set t-batch range if different; trigger resample & set base_samples to None
        if not hasattr(self, "_batch_range") or self.batch_range != batch_range:
            self.register_buffer("_batch_range", torch.tensor(batch_range, dtype=torch.long))
            self.register_buffer("base_samples", None)

    def forward(self, mean, variance) -> Tensor:
        r"""Draws MC samples from the posterior."""
        mvn = MultivariateNormal(
            torch.Tensor(np.array(mean)).unsqueeze(0),
            torch.Tensor([np.diag(variance[i, :]) for i in range(variance.shape[0])]).unsqueeze(0),
        )
        return mvn.rsample(sample_shape=self._sample_shape)

    @property
    def sample_shape(self) -> torch.Size:
        r"""The shape of a single sample."""
        return self._sample_shape
