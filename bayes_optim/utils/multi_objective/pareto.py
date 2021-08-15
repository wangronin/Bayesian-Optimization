from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


# TODO: base `non_dominated_set_2d` and `fast_non_dominated_sort` on Tensor
def non_dominated_set_2d(y, minimize=True):
    """
    Argument
    --------
    y : numpy 2d array,
        where the each solution occupies a row
    """
    y = np.asarray(y)
    N, _ = y.shape

    if isinstance(minimize, bool):
        minimize = [minimize]

    minimize = np.asarray(minimize).ravel()
    assert len(minimize) == 1 or minimize.shape == (N,)
    y *= (np.asarray([-1] * N) ** minimize).reshape(-1, 1)

    _ = np.argsort(y[:, 0])[::-1]
    y2 = y[_, 1]
    ND = []
    for i in range(N):
        v = y2[i]
        if not any(v <= y2[ND]) or len(ND) == 0:
            ND.append(i)
    return _[ND]


def fast_non_dominated_sort(fitness):
    fronts = []
    dominated_set = []
    mu = fitness.shape[1]
    n_domination = np.zeros(mu)

    for i in range(mu):
        p = fitness[:, i]
        p_dominated_set = []
        n_p = 0

        for j in range(mu):
            q = fitness[:, j]
            if i != j:
                # TODO: verify this part
                # check the strict domination
                # allow for duplication points on the same front
                if all(p <= q) and not all(p == q):
                    p_dominated_set.append(j)
                elif all(p >= q) and not all(p == q):
                    n_p += 1

        dominated_set.append(p_dominated_set)
        n_domination[i] = n_p

    # create the first front
    fronts.append(np.nonzero(n_domination == 0)[0].tolist())
    n_domination[n_domination == 0] = -1

    i = 0
    while True:
        for p in fronts[i]:
            p_dominated_set = dominated_set[p]
            n_domination[p_dominated_set] -= 1

        _front = np.nonzero(n_domination == 0)[0].tolist()
        n_domination[n_domination == 0] = -1

        if len(_front) == 0:
            break
        fronts.append(_front)
        i += 1

    return fronts


def is_non_dominated(Y: Tensor, deduplicate: bool = True) -> Tensor:
    r"""Computes the non-dominated front.

    Note: this assumes maximization.

    Args:
        Y: A `(batch_shape) x n x m`-dim tensor of outcomes.
        deduplicate: A boolean indicating whether to only return
            unique points on the pareto frontier.

    Returns:
        A `(batch_shape) x n`-dim boolean tensor indicating whether
        each point is non-dominated.
    """
    if Y.shape[-2] == 0:
        return torch.zeros(Y.shape[:-1], dtype=torch.bool, device=Y.device)
    Y1 = Y.unsqueeze(-3)
    Y2 = Y.unsqueeze(-2)
    dominates = (Y1 >= Y2).all(dim=-1) & (Y1 > Y2).any(dim=-1)
    nd_mask = ~(dominates.any(dim=-1))
    if deduplicate:
        # remove duplicates
        # find index of first occurrence  of each unique element
        indices = (Y1 == Y2).all(dim=-1).long().argmax(dim=-1)
        keep = torch.zeros_like(nd_mask)
        keep.scatter_(dim=-1, index=indices, value=1.0)
        return nd_mask & keep
    return nd_mask
