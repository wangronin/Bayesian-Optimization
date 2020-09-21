# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 01:29:44 2018

@author: wangronin
"""

import cython
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double sqrt(double x)
    double exp(double x)

def corr_grad_theta(theta, X, R, nu=1.5, corr_type='matern'):
    # Check input shapes
    X = np.atleast_2d(X)
    cdef int n_eval = X.shape[0]
    cdef int n_features = X.shape[1]

    diff = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2.

    if corr_type == 'squared_exponential':
        grad = -diff * R[..., np.newaxis]

    elif corr_type == 'matern':
        c = sqrt(3)
        D = np.sqrt(np.sum(theta * diff, axis=-1))

        if nu == 0.5:
            grad = - diff * theta / D * R
        elif nu == 1.5:
            grad = -3 * np.exp(-c * D)[..., np.newaxis] * diff / 2.
        elif nu == 2.5:
            pass

    elif corr_type == 'absolute_exponential':
        grad = -sqrt(diff) * R[..., np.newaxis]
    elif corr_type == 'generalized_exponential':
        pass
    elif corr_type == 'cubic':
        pass
    elif corr_type == 'linear':
        pass

    return grad
    
def matern(theta, X, eval_Dx=False, eval_Dtheta=False,
           length_scale_bounds=(1e-5, 1e5), nu=1.5):
    """
    theta = np.asarray(theta, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    if d.ndim > 1:
        n_features = d.shape[1]
    else:
        n_features = 1

    if theta.size == 1:
        return np.exp(-theta[0] * np.sum(d ** 2, axis=1))
    elif theta.size != n_features:
        raise ValueError("Length of theta must be 1 or %s" % n_features)
    else:
        return np.exp(-np.sum(theta.reshape(1, n_features) * d ** 2, axis=1))

    """
    theta = np.asarray(theta, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim > 1:
        n_features = X.shape[1]
    else:
        n_features = 1

    if theta.size == 1:
        dists = np.sqrt(theta[0] * np.sum(X ** 2, axis=1))
    else:
        dists = np.sqrt(np.sum(theta.reshape(1, n_features) * X ** 2, axis=1))

    # Matern 1/2
    if nu == 0.5:
        K = np.exp(-dists)
    # Matern 3/2
    elif nu == 1.5:

        K = dists * np.sqrt(3)
        K = (1. + K) * np.exp(-K)
    # Matern 5/2
    elif nu == 2.5:
        K = dists * np.sqrt(5)
        K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
    else:  # general case; expensive to evaluate
        pass

    return K