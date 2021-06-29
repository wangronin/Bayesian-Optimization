import logging
import os
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from .acquisition_fun import EI, MGFI, PI, UCB
from .bayes_opt import BO, AnnealingBO, NoisyBO, ParallelBO
from .mobo import MOBO
from .search_space import (
    Bool,
    BoolSpace,
    Discrete,
    DiscreteSpace,
    Integer,
    IntegerSpace,
    Ordinal,
    Real,
    RealSpace,
    SearchSpace,
)
from .solution import Solution
from .surrogate import GaussianProcess, RandomForest, trend

__all__ = [
    "BO",
    "ParallelBO",
    "NoisyBO",
    "AnnealingBO",
    "MOBO",
    "Solution",
    "RandomForest",
    "GaussianProcess",
    "trend",
    "SearchSpace",
    "IntegerSpace",
    "RealSpace",
    "BoolSpace",
    "DiscreteSpace",
    "EI",
    "UCB",
    "PI",
    "MGFI",
    "RandomForest",
    "fmin",
    "Integer",
    "Ordinal",
    "Real",
    "Bool",
    "Discrete",
]

# To use `dill` for the pickling, which works for
# much more python objects
os.environ["LOKY_PICKLER"] = "dill"

verbose = {False: logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO}

Vector = List[float]
Matrix = List[Vector]

# TODO: implement the code for `callback`, `xtol`, `ftol`, and `kwargs`
def fmin(
    func: Callable,
    lower: Union[float, Vector],
    upper: Union[float, Vector],
    x0: Union[int, Matrix, np.ndarray, None] = None,
    y0: Union[Vector, None] = None,
    n_point: int = 1,
    args: Tuple = (),
    xtol: float = 1e-4,
    ftol: float = 1e-4,
    max_FEs: Optional[int] = None,
    verbose: Optional[bool] = False,
    callback: Optional[Callable] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[Vector, float, int, int, List[np.ndarray]]:
    """Minimize a function using the Bayesian Optimization algorithm, which only uses
    function values, not derivatives or second derivatives. This function maintains an
    interface similar to `scipy.optimize.fmin`. Hereafter, we use the following customized
    types to describe the usage:

    - Vector = List[float]
    - Matrix = List[Vector]

    Parameters
    ----------
    func : Callable
        The objective function to be minimized.
    lower : Union[float, Vector]
        The lower bound of search variables, from which the search dimension is inferred.
        When it is not a `float`, it must have the same length as `upper`.
    upper : Union[float, Vector]
        The upper bound of search variables, from which the search dimension is inferred.
        When it is not a `float`, it must have the same length as `lower`.
    x0 : Union[int, Matrix, None], optional
        When it takes integer values, it specifies the number of initial sample points to
        take; When it is a 2d numpy array (or a nested list that contains the same data),
        it provides the initial sample points.
    y0 : Union[Vector, None], optional
        When `x0` is a 2d numpy array, it contains the function values pertaining to `x0`;
        Otherwise, it is ignored. It is by default None.
    n_point : int, optional
        The number of trial points generated in each iteration, by default 1
    args : Tuple, optional
        Extra arguments passed to `func`, i.e., ``func(x, *args)``.
    xtol : float, optional
        Absolute error in xopt between iterations that is acceptable for convergence,
        by default 1e-4.
    ftol : float, optional
        Absolute error in func(xopt) between iterations that is acceptable for convergence,
        by default 1e-4.
    max_FEs : Optional[int], optional
        Maximal number of function evaluations to make, by default None.
    verbose : Optional[bool], optional
        Verbosity of the output, by default False.
    callback : Optional[Callable], optional
        Called after each iteration, as `callback(X)`, where `X` is the current parameter
        vectors, by default None.
    seed : Optional[int], optional
        Seeding the random number generator in `numpy`, by default None.

    Returns
    -------
    Tuple[Vector, float, int, int, List[np.ndarray]]
        The return value contains:
        - The best-so-far parameter vector
        - The best-so-far function value
        - The number of iterations
        - The number of function evaluations
        - A list of trial points/vectors generated in every iteration

    Examples
    --------
    >>> def f(x):
    ...     return sum(x ** 2)
    >>> from bayes_optim import fmin
    >>>
    >>> minimum = fmin(f, [-5] * 2, [5] * 2, max_FEs=30, seed=42)
    Optimization terminated successfully.
            Current function value: 0.007165794451494286
            Iterations: 21
            Function evaluations: 30
    >>> minimum[0]
    0.007165794451494286
    """
    if seed is not None:
        _state = np.random.get_state()
        np.random.seed(seed)

    obj_func = lambda x: func(x, *args)

    if isinstance(lower, float) and isinstance(upper, float):
        dim = 1
        space = RealSpace([lower, upper])
    elif isinstance(lower, list) and isinstance(upper, list):
        assert len(lower) == len(upper)
        dim = len(lower)
        space = RealSpace(list(zip(lower, upper)))
        lower, upper = np.array(lower), np.array(upper)

    mean = trend.constant_trend(dim, beta=0)
    thetaL = 1e-10 * (upper - lower) * np.ones(dim)
    thetaU = 10 * (upper - lower) * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL
    model = GaussianProcess(
        mean=mean,
        corr="squared_exponential",
        theta0=theta0,
        thetaL=thetaL,
        thetaU=thetaU,
        nugget=0,
        noise_estim=False,
        optimizer="BFGS",
        wait_iter=3,
        random_start=dim,
        likelihood="concentrated",
        eval_budget=100 * dim,
    )

    # set up the warm-starting and DoE size
    if isinstance(x0, int):
        DoE_size = x0
        warm_data = ()
    elif hasattr(x0, "__iter__"):
        DoE_size = None
        if y0 is None:
            y0 = [obj_func(_) for _ in x0]
        warm_data = (x0, y0)
    else:
        DoE_size = None
        warm_data = ()

    _BO = BO if n_point == 1 else ParallelBO
    opt = _BO(
        search_space=space,
        obj_fun=obj_func,
        model=model,
        DoE_size=DoE_size,
        warm_data=warm_data,
        eval_type="list",
        max_FEs=max_FEs,
        verbose=verbose,
        n_point=n_point,
    )
    opt.run()

    if seed is not None:
        np.random.set_state(_state)

    N, n = opt.DoE_size, opt.n_point
    _data, data = opt.data[:N, :], opt.data[N:, :]

    data_per_iteration = [np.asarray(data[:N, :])]
    data_per_iteration += [
        np.asarray(data[(i * n) : ((i + 1) * n), :]) for i in range(opt.iter_count - 1)
    ]

    print(
        "Optimization terminated successfully.\n"
        "        Current function value: {}\n"
        "        Iterations: {}\n"
        "        Function evaluations: {}\n".format(
            opt.xopt.fitness, opt.iter_count, opt.eval_count
        )
    )
    return opt.xopt, opt.xopt.fitness, opt.iter_count, opt.eval_count, data_per_iteration
