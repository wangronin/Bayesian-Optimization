# -*- coding: utf-8 -*-
"""
Created on Mon May 19 10:17:43 2014

@author: Hao Wang
@email: wangronin@gmail.com
"""
from __future__ import print_function

import pdb
import logging
from copy import copy

import numpy as np
from numpy import isfinite, mod, floor, shape, bitwise_and, zeros, newaxis

# TODO: re-written those functions to C/Cython
def non_dominated_set_2d(y, minimize=True):
    """ 
    Argument
    --------
    y : numpy 2d array, 
        where the each solution occupies a row
    """
    y = np.asarray(y)
    N, dim = y.shape

    if isinstance(minimize, bool):
        minimize = [minimize]
    
    minimize = np.asarray(minimize).ravel()
    assert len(minimize) == 1 or minimize.shape == (N, ) 
    y *= (np.asarray([-1] * N) ** minimize).reshape(-1, 1)

    _ = np.argsort(y[:, 0])[::-1]
    y2 = y[_, 1]
    ND = []
    for i in range(N):
        v = y2[i]
        if not any(v <= y2[ND]) or len(ND) == 0:
            ND.append(i) 
    return _[ND]

def non_dominated_set_3d(y, minimize=True):
    pass

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
        idx = list(range(0, len(perf)))
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
    x = np.asarray(x, dtype='float')
    shape_ori = x.shape
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
    return x.reshape(shape_ori)

if __name__ == '__main__':
    # TODO: goes to unittest
    np.random.seed(1)
    perf = np.random.randn(20)
    print(perf)
    print(proportional_selection(perf, 20, minimize=False, replacement=False))