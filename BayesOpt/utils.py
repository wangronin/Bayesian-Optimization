# -*- coding: utf-8 -*-
"""
Created on Mon May 19 10:17:43 2014

@author: Hao Wang
@email: wangronin@gmail.com
"""
import pdb
from copy import copy
import numpy as np
from numpy import isfinite, mod, floor, shape, bitwise_and, zeros, newaxis

# TODO: implement this as a C procedure
def proportional_selection(perf, N, minimize=True, replacement=True):
    def select(perf):
        perf_min = np.min(perf)
        interval = np.cumsum((perf - perf_min) / (np.sum(perf) - perf_min * len(perf)))
        return np.nonzero(np.random.rand() <= interval)[0][0]
    
    perf = np.array(perf)
    if minimize:
        perf = -perf
        perf -= np.min(perf)

    if replacement:
        res = [select(perf) for i in range(N)]
    else:
        assert N <= len(perf)
        perf_ = copy(perf)
        idx = range(0, len(perf))
        res = []
        for i in range(N):
            if len(perf_) == 1:
                res.append(idx[0])
            else:
                _ = select(perf_)
                res.append(idx[_])
                perf_ = np.delete(perf_, _)
                del idx[_]
    return res

# TODO: double check this one. It causes the explosion of step-sizes in MIES
def boundary_handling(x, lb, ub):
    """
    
    This function transforms x to t w.r.t. the low and high
    boundaries lb and ub. It implements the function T^{r}_{[a,b]} as
    described in Rui Li's PhD thesis "Mixed-Integer Evolution Strategies
    for Parameter Optimization and Their Applications to Medical Image 
    Analysis" as alorithm 6.
    
    """
    x = np.atleast_2d(x)
    lb = np.atleast_1d(lb)
    ub = np.atleast_1d(ub)
    
    transpose = False
    if x.shape[0] != len(lb):
        x = x.T
        transpose = True
    
    lb, ub = lb.flatten(), ub.flatten()
    
    lb_index = isfinite(lb)
    up_index = isfinite(ub)
    
    valid = bitwise_and(lb_index,  up_index)
    
    LB = lb[valid][:, newaxis]
    UB = ub[valid][:, newaxis]

    y = (x[valid, :] - LB) / (UB - LB)
    I = mod(floor(y), 2) == 0
    yprime = zeros(shape(y))
    yprime[I] = np.abs(y[I] - floor(y[I]))
    yprime[~I] = 1.0 - np.abs(y[~I] - floor(y[~I]))

    x[valid, :] = LB + (UB - LB) * yprime
    
    if transpose:
        x = x.T
    return x

if __name__ == '__main__':
    np.random.seed(1)
    perf = np.random.randn(20)
    print perf
    print proportional_selection(perf, 20, minimize=False, replacement=False)