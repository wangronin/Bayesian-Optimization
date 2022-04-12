from typing import Callable, Dict

# default aquisition optimization budget
# TODO: create an Option class similar to CMA-ES
default_AQ_max_FEs: Dict = {
    "MIES": lambda dim: int(1e3 * dim),
    "BFGS": lambda dim: int(1e2 * dim),
    "OnePlusOne_Cholesky_CMA": lambda dim: int(1e3 * dim),
    "DE": lambda dim: int(1e3 * dim),
}

default_AQ_n_restart: Callable = lambda dim: max(5, int(dim / 2))
default_AQ_wait_iter: int = 3
