# -*- coding: utf-8 -*-
"""
Created on Mon May 19 10:17:43 2014

@author: wangronin
"""

import numpy as np
from numpy import isfinite, mod, floor, shape, bitwise_and, zeros, newaxis


def boundary_handling(x, lb, ub):
    """

    This function transforms x to t w.r.t. the low and high
    boundaries lb and ub. It implements the function T^{r}_{[a,b]} as
    described in Rui Li's PhD thesis "Mixed-Integer Evolution Strategies
    for Parameter Optimization and Their Applications to Medical Image
    Analysis" as alorithm 6.

    """

    lb, ub = lb.flatten(), ub.flatten()

    lb_index = isfinite(lb)
    up_index = isfinite(ub)

    valid = bitwise_and(lb_index, up_index)

    LB = lb[valid][:, newaxis]
    UB = ub[valid][:, newaxis]

    y = (x[valid, :] - LB) / (UB - LB)
    I = mod(floor(y), 2) == 0
    yprime = zeros(shape(y))
    yprime[I] = np.abs(y[I] - floor(y[I]))
    yprime[~I] = 1.0 - np.abs(y[~I] - floor(y[~I]))

    x[valid, :] = LB + (UB - LB) * yprime

    return x
