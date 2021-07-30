import logging
import time
from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from ..search_space import RealSpace
from ..utils import dynamic_penalty
from .mies import MIES
from .one_plus_one_cma_es import OnePlusOne_Cholesky_CMA, OnePlusOne_CMA

__all__ = ["argmax_restart", "OnePlusOne_CMA", "OnePlusOne_Cholesky_CMA", "MIES"]


def finite_difference(f: Callable, x: np.ndarray, delta: float = 1e-5, **kwargs):
    N = len(x)
    v = np.eye(N) * delta
    return np.array(
        [(f(x + v[i], **kwargs) - f(x - v[i], **kwargs)) / (2 * delta) for i in range(N)]
    ).reshape(-1, 1)


class Penalized:
    """Construct a penalized problem of form, `f(x) + p(x)`, where `p` is
    the `dynamic_penalty` function from ..utils module. We also take care of approximating
    the gradient, which is then used by the BFGS algorithm.
    """

    def __init__(self, func: Callable, h: Callable, g: Callable):
        self.func = func
        self.h = h
        self.g = g
        self.t = 10

    def __call__(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        f, fg = tuple(map(lambda x: -1.0 * x, self.func(x)))
        if self.h or self.g:
            p = dynamic_penalty(x, t=self.t, equality=self.h, inequality=self.g)
            pg = finite_difference(
                dynamic_penalty, x, t=self.t, equality=self.h, inequality=self.g
            )
            self.t += 1
            return f + p, fg + pg
        return f, fg


def argmax_restart(
    obj_func: Callable,
    search_space,
    h: Callable = None,
    g: Callable = None,
    eval_budget: int = 100,
    n_restart: int = 10,
    wait_iter: int = 3,
    optimizer: str = "BFGS",
    logger: logging.Logger = None,
):
    # lists of the best solutions and acquisition values from each restart
    xopt, fopt = [], []
    best = -np.inf
    wait_count = 0
    if not isinstance(search_space, RealSpace) and optimizer == "BFGS":
        optimizer = "MIES"
        logger.warning("L-BFGS-B can only be applied on continuous search space")

    for iteration in range(n_restart):
        x0 = search_space.sample(N=1, method="uniform")[0]
        if optimizer == "BFGS":
            bounds = np.array(search_space.bounds)
            # TODO: this is still subject to testing
            xopt_, fopt_, stop_dict = fmin_l_bfgs_b(
                Penalized(obj_func, h, g),
                x0,
                pgtol=1e-8,
                factr=1e6,
                bounds=bounds,
                maxfun=eval_budget,
            )
            xopt_ = xopt_.flatten().tolist()
            if not isinstance(fopt_, float):
                fopt_ = float(fopt_)
            fopt_ = -fopt_

            if logger is not None and stop_dict["warnflag"] != 0:
                logger.debug("L-BFGS-B terminated abnormally with the state: %s" % stop_dict)

        elif optimizer == "OnePlusOne_Cholesky_CMA":
            opt = OnePlusOne_Cholesky_CMA(
                search_space=search_space,
                obj_fun=obj_func,
                h=h,
                g=g,
                max_FEs=eval_budget,
                ftol=1e-4,
                xtol=1e-4,
                n_restart=0,
                minimize=False,
                verbose=False,
                seed=time.time(),
            )
            xopt_, fopt_, stop_dict = opt.run()
            stop_dict["funcalls"] = stop_dict["FEs"] if "FEs" in stop_dict else opt.eval_count

        elif optimizer == "MIES":
            xopt_, fopt_, stop_dict = MIES(
                search_space,
                obj_func,
                eq_func=h,
                ineq_func=g,
                max_eval=eval_budget,
                minimize=False,
                verbose=False,
                eval_type="list",
            ).optimize()

        cond_h = all(np.isclose(np.abs(h(xopt_)), 0, atol=1e-1)) if h else True
        cond_g = all(g(xopt_) <= 0) if g else True
        if cond_h and cond_g:
            if isinstance(fopt_, np.ndarray):
                fopt_ = fopt_[0]

            if fopt_ > best:
                best = fopt_
                wait_count = 0

                if logger is not None:
                    logger.debug(
                        "restart : %d - funcalls : %d - Fopt : %f"
                        % (iteration + 1, stop_dict["funcalls"], fopt_)
                    )
            else:
                wait_count += 1

            eval_budget -= stop_dict["funcalls"]
            xopt.append(xopt_)
            fopt.append(fopt_)

        if eval_budget <= 0 or wait_count >= wait_iter:
            break

    if len(xopt) == 0:
        return [], []

    idx = np.argsort(fopt)[::-1]
    return xopt[idx[0]], fopt[idx[0]]
