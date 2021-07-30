import functools
import time
from typing import Callable, Dict, List, Union

import numpy as np

from ._exception import ConstraintEvaluationError
from .solution import Solution


def is_pareto_efficient(fitness, return_mask: bool = True) -> Union[List[int], List[bool]]:
    """get the Pareto efficient subset

    Parameters
    ----------
    fitness : np.ndarray of shape (n_points, n_obj)
        the objective value
    return_mask : bool, optional
        if returning a mask, by default True

    Returns
    -------
    An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """

    is_efficient = np.arange(fitness.shape[0])
    n_points = fitness.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(fitness):
        nondominated_point_mask = np.any(fitness < fitness[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        fitness = fitness[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    return is_efficient


def fillin_fixed_value(X: List[List], fixed: Dict, search_space):
    if fixed is None:
        return X
    if len(X) == 0:
        return X
    mask = np.array([v in fixed.keys() for v in search_space.var_name])
    values = [fixed[k] for i, k in enumerate(search_space.var_name) if mask[i]]
    out = np.empty((len(X), len(mask)), dtype=object)
    out[:, mask] = values
    out[:, ~mask] = X
    return out.tolist()


def partial_argument(
    func: callable,
    var_name: List[str],
    fixed: Dict[str, Union[str, float, int, object, bool]] = None,
    reduce_output: bool = False,
):
    """fill-in the values for inactive variables

    Parameters
    ----------
    func : callable
        the target function to call which is defined on the original search space
    masks : np.ndarray
        the mask array indicating which variables are deemed inactive
    values : np.ndarray
        the values fixed for the inactive variables
    """
    fixed = {} if fixed is None else fixed
    masks = np.array([v in fixed.keys() for v in var_name])
    values = [fixed[k] for i, k in enumerate(var_name) if masks[i]]

    @functools.wraps(func)
    def wrapper(X: Union[np.ndarray, Solution, list]):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=object)

        N = 1 if len(X.shape) == 1 else X.shape[1]
        X_ = np.empty((N, len(masks)), dtype=object)
        X_[:, masks] = values
        X_[:, ~masks] = X
        out_ = func(X_)

        # TODO: fix this ad-hoc solution for acquisition functions
        if reduce_output:
            out = []
            for v in tuple(out_):
                if isinstance(v, np.ndarray):
                    if len(v.shape) == 1 and len(v) > 1:
                        v = v[~masks]
                    elif len(v.shape) == 2:
                        if v.shape[0] == len(masks):
                            v = v[~masks, :]
                        elif v.shape[1] == len(masks):
                            v = v[:, ~masks]
                elif isinstance(v, list) and len(v) == len(masks):
                    v = [v[m] for m in ~masks]
                out.append(v)
            return tuple(out)
        return out_

    return wrapper


def func_with_list_arg(func, arg_type, var_names):
    @functools.wraps(func)
    def wrapper(X):
        if isinstance(X, (list, tuple)):
            X = np.array(X, dtype="object")
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        X = Solution(X, var_name=var_names)
        if arg_type == "list":
            X = X.tolist()
        elif arg_type == "dict":
            X = X.to_dict()
        return np.array([func(_) for _ in X]).ravel()

    return wrapper


def timeit(func):
    @functools.wraps(func)
    def __func__(ref, *arg, **kwargv):
        t0 = time.time()
        out = func(ref, *arg, **kwargv)
        if hasattr(ref, "logger"):
            ref.logger.info(f"{func.__name__} takes {time.time() - t0:.4f}s")
        else:
            print(f"{func.__name__} takes {time.time() - t0:.4f}s")
        return out

    return __func__


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
    elif hasattr(bound, "__iter__"):
        bound = list(bound)
        if len(bound) == 1:
            bound *= dim
    assert len(bound) == dim
    return np.asarray(bound)


def dynamic_penalty(
    X: List,
    t: int = 1,
    equality: Callable = None,
    inequality: Callable = None,
    C: float = 0.5,
    alpha: float = 1,
    beta: float = 1.5,
    epsilon: float = 1e-1,
    minimize: bool = True,
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
    if not hasattr(X[0], "__iter__") or isinstance(X[0], str):
        X = [X]
    X = np.array(X, dtype=object)

    N = len(X)
    p = np.zeros(N)

    if equality is not None:
        try:
            v = np.atleast_2d(list(map(equality, X))).reshape(N, -1)
        except Exception as e:
            raise ConstraintEvaluationError(X, str(e)) from None
        v[np.abs(v) <= epsilon] = 0
        p += np.sum(np.abs(v), axis=1)

    if inequality is not None:
        try:
            v = np.atleast_2d(list(map(inequality, X))).reshape(N, -1)
        except Exception as e:
            raise ConstraintEvaluationError(X, str(e)) from None
        # NOTE: inequalities are always tested with less or equal relation.
        # Inequalities with strict less conditions should be created by adding a tiny epsilon
        # to the constraint
        v[v <= 0] = 0
        p += np.sum(np.abs(v) ** beta, axis=1)

    p = (C * t) ** alpha * p * (-1) ** (not minimize)
    return p
