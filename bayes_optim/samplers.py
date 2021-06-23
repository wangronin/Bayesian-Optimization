r"""
Sampler modules to be used with MC-evaluated acquisition functions.
"""

from __future__ import annotations

from abc import ABC
from typing import Tuple

import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
from torch.nn import Module


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
