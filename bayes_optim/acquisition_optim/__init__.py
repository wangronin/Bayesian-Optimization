import logging
from typing import Any, Callable, List, Tuple, Union

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from ..search_space import RealSpace
from .mies import MIES
from .one_plus_one_cma_es import OnePlusOne_Cholesky_CMA, OnePlusOne_CMA

__all__ = ["argmax_restart", "OnePlusOne_CMA", "OnePlusOne_Cholesky_CMA", "MIES"]


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
        logger.warning("L-BFGS-B cannot be applied on continuous search space")

    if (h is not None or g is not None) and optimizer == "BFGS":
        optimizer = "OnePlusOne_Cholesky_CMA"
        # TODO: add constraint handling for BFGS
        logger.warning("L-BFGS-B cannot be applied with constraints at this moment")

    for iteration in range(n_restart):
        x0 = search_space.sample(N=1, method="uniform")[0]

        if optimizer == "BFGS":
            bounds = np.array(search_space.bounds)

            if not all([isinstance(_, float) for _ in x0]):
                raise ValueError("BFGS is not supported with mixed variable types.")

            func = lambda x: tuple(map(lambda x: -1.0 * x, obj_func(x)))
            xopt_, fopt_, stop_dict = fmin_l_bfgs_b(
                func, x0, pgtol=1e-8, factr=1e6, bounds=bounds, maxfun=eval_budget
            )

            xopt_ = xopt_.flatten().tolist()
            if not isinstance(fopt_, float):
                fopt_ = float(fopt_)
            fopt_ = -fopt_

            if logger is not None and stop_dict["warnflag"] != 0:
                logger.debug("L-BFGS-B terminated abnormally with the state: %s" % stop_dict)
        elif optimizer == "OnePlusOne_Cholesky_CMA":
            lb, ub = list(zip(*search_space.bounds))
            opt = OnePlusOne_Cholesky_CMA(
                dim=search_space.dim,
                obj_fun=obj_func,
                h=h,
                g=g,
                lb=lb,
                ub=ub,
                max_FEs=eval_budget,
                ftol=1e-4,
                xtol=1e-4,
                n_restart=0,
                minimize=False,
                verbose=False,
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

    idx = np.argsort(fopt)[::-1]
    return xopt[idx[0]], fopt[idx[0]]
