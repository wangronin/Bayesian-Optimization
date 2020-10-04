import numpy as np
import atexit, signal, errno, os, sys, time

def set_bounds(bound, dim):
    if isinstance(bound, str):
        bound = eval(bound)
    elif isinstance(bound, (float, int)):
        bound = [bound] * dim
    elif hasattr(bound, '__iter__'):
        bound = list(bound)
        if len(bound) == 1:
            bound *= dim
    assert len(bound) == dim
    return np.asarray(bound)

def arg_to_int(arg):
    if isinstance(arg, str):
        x = int(eval(arg))
    elif isinstance(arg, (int, float)):
        x = int(arg)
    else:
        raise ValueError
    return x

# TODO: re-written those functions to C/Cython
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

def dynamic_penalty(X, t, equality=None, inequality=None, C=0.5, alpha=1, beta=2,
                    epsilon=0.01, minimize=True):
    X = np.atleast_2d(X)
    N = X.shape[0]
    p = np.zeros(N)

    if equality is not None:
        v = np.atleast_2d(list(map(equality, X))).reshape(N, -1)
        v[np.abs(v) <= epsilon] = 0
        p += np.sum(np.abs(v), axis=1)

    if inequality is not None:
        v = np.atleast_2d(list(map(inequality, X))).reshape(N, -1)
        v[v <= 0] = 0
        p += np.sum(np.abs(v) ** beta, axis=1)

    p = (-1) ** (not minimize) * (C * t) ** alpha * p
    return p

# TODO: get this done and test it
def stochastic_ranking(X, fitness, equality=None, inquality=None, P=0.4, gamma=1,
                       beta=1, epsilon=0):
    N = len(X) if isinstance(X, list) else X.shape[0]
    #N = X.shape[0]
    p = np.zeros(N)

    if equality is not None:
        v = np.atleast_2d(list(map(equality, X))).reshape(N, -1)
        v[np.abs(v) <= epsilon] = 0
        p += np.sum(np.abs(v) ** gamma, axis=1)

    if inquality is not None:
        v = np.atleast_2d(list(map(inquality, X))).reshape(N, -1)
        v[v <= 0] = 0
        p += np.sum(np.abs(v) ** beta, axis=1)