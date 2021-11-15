import logging
from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed

from .search_space import SearchSpace
from .solution import Solution
from .utils import get_logger, timeit
from .utils.exception import RecommendationUnavailableError


class BaseOptimizer:
    def __init__(
        self,
        search_space: SearchSpace,
        n_obj: int = 1,
        obj_fun: Optional[Callable] = None,
        parallel_obj_fun: Optional[Callable] = None,
        eq_fun: Optional[Callable] = None,
        ineq_fun: Optional[Callable] = None,
        n_job: int = 1,
        ftarget: Optional[float] = None,
        max_FEs: Optional[int] = None,
        minimize: bool = True,
        verbose: bool = False,
        log_file: Optional[str] = None,
        random_seed: Optional[int] = None,
        instance_id: Optional[str] = None,
    ):
        self.search_space: SearchSpace = search_space
        self.n_obj: int = n_obj
        self.random_seed: int = random_seed
        self.instance_id: str = instance_id if instance_id else str(id(self))

        self.obj_fun: callable = obj_fun
        self.parallel_obj_fun: callable = parallel_obj_fun
        self.h: callable = eq_fun
        self.g: callable = ineq_fun
        self.n_job: int = max(1, int(n_job))
        self.ftarget: float = ftarget
        self.minimize: bool = minimize
        self.verbose: bool = verbose
        self.max_FEs: int = int(max_FEs) if max_FEs else np.inf

        # self.data: Solution = None
        self.iter_count: int = 0
        self.eval_count: int = 0
        self.stop_dict: Dict[str, bool] = {}
        self.hist_f: List[float] = []

        self._to_pheno = lambda x: x
        self._to_geno = lambda x: x
        self.logger: logging.Logger = get_logger(
            logger_id=f"{self.__class__.__name__} ({self.instance_id})",
            file=log_file,
            console=verbose,
        )

    @abstractmethod
    def ask(
        self, n_point: int = None, fixed: Dict[str, Union[float, int, str]] = None
    ) -> Union[List[list], List[dict]]:
        """Suggest a list of candidate solutions

        Parameters
        ----------
        n_point : int, optional
            the number of candidates to request, by default None
        fixed : Dict[str, Union[float, int, str]], optional
            a dictionary specifies the decision variables fixed and the value to which those
            are fixed, by default None

        Returns
        -------
        Union[List[list], List[dict]]
            the suggested candidates
        """

    @abstractmethod
    def tell(
        self,
        X: List[Union[list, dict]],
        func_vals: List[Union[float, list]],
        h_vals: List[Union[float, list]] = None,
        g_vals: List[Union[float, list]] = None,
        index: List[str] = None,
        warm_start: bool = False,
    ):
        """Tell the BO about the function values of proposed candidate solutions

        Parameters
        ----------
        X : List of Lists or Solution
            The candidate solutions which are usually proposed by the `self.ask` function
        func_vals : List/np.ndarray of reals
            The corresponding function values
        """

    def recommend(self) -> Solution:
        if self.xopt is None or len(self.xopt) == 0:
            raise RecommendationUnavailableError()
        return self.xopt

    @abstractmethod
    def step(self):
        """Implement one step/iteration of the optimization process"""

    @timeit
    def evaluate(self, X) -> List[float]:
        """Evaluate the candidate points and update evaluation info in the dataframe"""
        # Parallelization is handled by the objective function itself
        if self.parallel_obj_fun is not None:
            func_vals = self.parallel_obj_fun(X)
        else:
            if self.n_job > 1:  # or by ourselves..
                func_vals = Parallel(n_jobs=self.n_job)(delayed(self.obj_fun)(x) for x in X)
            else:  # or sequential execution
                func_vals = [self.obj_fun(x) for x in X]
        return func_vals

    def check_stop(self) -> bool:
        if self.eval_count >= self.max_FEs:
            self.stop_dict["max_FEs"] = self.eval_count

        if self.ftarget is not None and self.xopt is not None:
            if self._compare(self.xopt.fitness[0], self.ftarget):
                self.stop_dict["ftarget"] = self.xopt.fitness[0]

        return bool(self.stop_dict)

    def run(self) -> Tuple[List[Solution], dict]:
        while not self.check_stop():
            self.step()
        return self._to_pheno(self.xopt), self.xopt.fitness, self.stop_dict

    @property
    def xopt(self) -> Solution:
        if not hasattr(self, "data"):
            return None
        fopt = self._get_best(self.data.fitness)
        self._xopt = self.data[np.where(self.data.fitness == fopt)[0][0]]
        return self._xopt

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed):
        if seed:
            self._random_seed = int(seed)
            if self._random_seed:
                np.random.seed(self._random_seed)

    @property
    def search_space(self):
        return self._search_space

    @search_space.setter
    def search_space(self, search_space):
        self._search_space = search_space
        self.dim = len(self._search_space)
        self.var_names = self._search_space.var_name
        self.r_index = self._search_space.real_id  # indices of continuous variable
        self.i_index = self._search_space.integer_id  # indices of integer variable
        self.d_index = self._search_space.categorical_id  # indices of categorical variable
        self.param_type = self._search_space.var_type
        self.N_r = len(self.r_index)
        self.N_i = len(self.i_index)
        self.N_d = len(self.d_index)

    def _get_best(self, fitness: np.ndarray) -> float:
        return np.min(fitness) if self.minimize else np.max(fitness)

    def _compare(self, f1: float, f2: float) -> bool:
        """Test if objecctive value f1 is better than f2"""
        return f1 < f2 if self.minimize else f2 > f1
