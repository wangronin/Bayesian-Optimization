import numpy as np

def dynamic_penalty(X, t, equality=None, inquality=None, C=0.5, alpha=1, beta=2, 
                    epsilon=0.01, minimize=True):
    N = len(X) if isinstance(X, list) else X.shape[0]
    # N = X.shape[0]
    p = np.zeros(N)

    if equality is not None:
        v = np.atleast_2d(list(map(equality, X))).reshape(N, -1)
        v[np.abs(v) <= epsilon] = 0
        p += np.sum(np.abs(v), axis=1)

    if inquality is not None:
        v = np.atleast_2d(list(map(inquality, X))).reshape(N, -1)
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
