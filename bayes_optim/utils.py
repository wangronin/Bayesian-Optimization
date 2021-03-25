from typing import Callable, List
import numpy as np

def arg_to_int(arg):
    if isinstance(arg, str):
        x = int(eval(arg))
    elif isinstance(arg, (int, float)):
        x = int(arg)
    else:
        raise ValueError
    return x

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

# TODO: move this to a '_penalty.py' file
def dynamic_penalty(
    X: List,
    t: int = 1,
    equality: Callable = None,
    inequality: Callable = None,
    C: float = 0.5,
    alpha: float = 1,
    beta: float = 2,
    epsilon: float = 1e-2,
    minimize: bool = True
) -> np.ndarray:
    r"""Dynamic Penalty calculated as follows:

    $$(tC)^{\alpha} * [\sum_i max(|h(x_i)|, \epsilon) + \sum_i max(0, g(x_i))^{\beta}],$$

    where $x_i$ -> each row of ``X``, h -> ``equality``, and g -> ``inequality``.

    TODO: give a reference here

    Parameters
    ----------
    X : np.ndarray
        Input candidate solutions
    t : int, optional
        The iteration number of the optimization algorithm employing this method, by default 1
    equality : Callable, optional
        Equality function, by default None
    inequality : Callable, optional
        Inequality function, by default None
    C : float, optional
        coefficient of the iteration term, by default 0.5
    alpha : float, optional
        exponent to the iteration term, by default 1
    beta : float, optional
        coefficient to the inequality terms, by default 2
    epsilon : float, optional
        threshold to determine whether the equality constraint is met, by default 1e-4
    minimize : bool, optional
        minimize or maximize? by default True

    Returns
    -------
    ``p``
        the dynamic penalty value
    """
    if not hasattr(X[0], '__iter__') or isinstance(X[0], str):
        X = [X]

    N = len(X)
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

# TODO: get this done and test it and add docstrings..
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
