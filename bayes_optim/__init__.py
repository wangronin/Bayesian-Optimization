import logging
import os
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from .acquisition.acquisition_fun import EI, MGFI, PI, UCB
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

__all__: List[str] = [
    "BO",
    "ParallelBO",
    "NoisyBO",
    "AnnealingBO",
    "MOBO",
    "Solution",
    "RandomForest",
    "GaussianProcess",
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
    "trend",
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
    max_FEs: Optional[int] = None,
    verbose: Optional[bool] = False,
    # callback: Optional[Callable] = None,
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
    max_FEs : Optional[int], optional
        Maximal number of function evaluations to make, by default None.
    verbose : Optional[bool], optional
        Verbosity of the output, by default False.
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
    obj_func = lambda x: func(x, *args)

    if isinstance(lower, float) and isinstance(upper, float):
        search_space = RealSpace([lower, upper])
    elif isinstance(lower, list) and isinstance(upper, list):
        assert len(lower) == len(upper)
        search_space = RealSpace(list(zip(lower, upper)))

    dim = search_space.dim
    bounds = np.asarray(search_space.bounds)
    model = GaussianProcess(
        mean=trend.constant_trend(dim),
        corr="matern",
        thetaL=1e-3 * (bounds[:, 1] - bounds[:, 0]),
        thetaU=1e3 * (bounds[:, 1] - bounds[:, 0]),
        nugget=1e-6,
        noise_estim=False,
        optimizer="BFGS",
        wait_iter=3,
        random_start=max(10, dim),
        likelihood="concentrated",
        eval_budget=100 * dim,
        random_state=seed,
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
        search_space=search_space,
        obj_fun=obj_func,
        model=model,
        DoE_size=DoE_size,
        warm_data=warm_data,
        eval_type="list",
        max_FEs=max_FEs,
        verbose=verbose,
        n_point=n_point,
        random_seed=seed,
        **kwargs,
    )
    opt.run()

    N, n = opt.DoE_size, opt.n_point
    _data, data = opt.data[:N, :], opt.data[N:, :]

    data_per_iteration = [np.asarray(data[:N, :])]
    data_per_iteration += [np.asarray(data[(i * n) : ((i + 1) * n), :]) for i in range(opt.iter_count - 1)]

    print(
        "Optimization terminated successfully.\n"
        "        Current function value: {}\n"
        "        Iterations: {}\n"
        "        Function evaluations: {}\n".format(opt.xopt.fitness, opt.iter_count, opt.eval_count)
    )
    return opt.xopt, opt.xopt.fitness, opt.iter_count, opt.eval_count, data_per_iteration
