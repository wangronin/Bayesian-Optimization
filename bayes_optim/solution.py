from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from tabulate import tabulate

__authors__ = ["Hao Wang"]

# TODO: maybe set var_name as a constant attribute
# TODO: register the objective function the class and implement an eval function
# TODO: fix the maximum recursion problem when debugging
# TODO: make this class work with numpy's ufunc, e.g., ``np.round``


class Solution(np.ndarray):
    """Subclassing numpy array to represent set of solutions in the optimization
    Goal to achieve:
     1) heterogenous data types, like pandas
     2) easy indexing as np.ndarray
     3) extra attributes (e.g., fitness) sliced together with the solution
    """

    def __new__(
        cls,
        x: Sequence,
        fitness: Optional[Sequence] = None,
        n_eval: Union[int, List[int]] = 0,
        index: Union[str, List[str]] = None,
        var_name: Union[str, List[str]] = None,
        fitness_name: Union[str, List[str]] = None,
        n_obj: int = 1,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        x : Sequence
            The parameters of a solution
        fitness : Optional[Sequence], optional
            objective values of solutions, by default None
        n_eval : Union[int, List[int]], optional
            number of evaluations per each solution, by default 0
        index : Union[int, List[int]], optional
            indices of solutions, by default None
        var_name : Union[str, List[str]], optional
            names of the variables, by default None
        fitness_name : Union[str, List[str]], optional
            names of the fitness values, by default None
        n_obj : int, optional
            the number of objective functions, by default 1
        verbose : bool, optional
            controls if additional information are printed when calling ``__str__``
                and to_dict, by default True

        Returns
        -------
        ``Solution``

        Note
        ----
            Instead of using `__init__`, the `__new__` function is used here because
            sometimes we would like to return an object of its subclasses, e.g., when
            slicing a subclass of `ndarray`, `ndarray.__new__(subclass, ...)` will
            return an object of type `subclass` while `ndarray.__init__(self, ...)`
            will return an object of `ndarray` (of course, `__init__` would work if
            the user also overloads the slicing function, which is not convenient).
            If attributes `index`, `fitness`, `n_eval` are modified in a slice of
            Solution, the corresponding attributes in the original object are also
            modified. Here, `var_name` is not affected by this behavior. This function
            is only called when explicitly constructing the `Solution` object.
            For slicing and view casting, the extra attributes are handled in function
            `__array_finalize__`.
        """
        obj = np.asarray(x, dtype="object").view(cls)

        if len(obj.shape) > 2:
            raise Exception("More than 2D is not supported")

        obj.N = 1 if len(obj.shape) == 1 else obj.shape[0]
        obj.dim = obj.shape[0] if len(obj.shape) == 1 else obj.shape[1]
        obj.n_obj = int(n_obj)

        # TODO: convert those as decorators
        if obj.n_obj > 1:  # multi-objective
            if not hasattr(fitness, "__iter__"):
                fitness = [[fitness] * obj.n_obj] * obj.N
            elif not hasattr(fitness[0], "__iter__"):
                assert len(fitness) == obj.n_obj
                fitness = [fitness] * obj.N
            assert all([len(_) == obj.n_obj for _ in fitness])
        elif obj.n_obj == 1:
            if not hasattr(fitness, "__iter__"):
                fitness = [fitness] * obj.N
        assert len(fitness) == obj.N

        if not hasattr(n_eval, "__iter__"):
            assert isinstance(n_eval, int)
            n_eval = [n_eval] * obj.N
        else:
            assert all(list(map(lambda n: isinstance(n, int), n_eval)))

        if index is None:
            index = list(map(str, range(obj.N)))
        elif isinstance(index, (str, int)):
            index = [index]
        assert len(index) == obj.N

        if var_name is None:
            if obj.dim == 1:
                var_name = ["x"]
            else:
                var_name = ["x" + str(i) for i in range(obj.dim)]
        assert len(var_name) == obj.dim

        if fitness_name is None:
            if obj.n_obj == 1:
                fitness_name = ["f"]
            else:
                fitness_name = ["f" + str(i) for i in range(obj.n_obj)]
        elif isinstance(fitness_name, str):
            assert obj.n_obj == 1
            fitness_name = [fitness_name]
        assert len(fitness_name) == obj.n_obj

        # TODO: a np.ndarray is used for those attributes because slicing it returns references
        super(Solution, obj).__setattr__("fitness", np.asarray(fitness, dtype=float))
        super(Solution, obj).__setattr__("n_eval", np.asarray(n_eval, dtype=int))
        super(Solution, obj).__setattr__("index", np.asarray(index, dtype=str))
        obj.var_name = np.asarray(var_name)
        obj.fitness_name = fitness_name
        obj.verbose = verbose
        return obj

    def __iadd__(self, other: Solution) -> Solution:
        return self.__add__(other)

    def __add__(self, other: Solution) -> Solution:
        """Concatenate two Solution objects"""
        assert isinstance(other, Solution)
        assert self.dim == other.dim
        assert self.n_obj == other.n_obj
        assert len(set(self.fitness_name).symmetric_difference(other.fitness_name)) == 0
        assert len(set(self.var_name).symmetric_difference(other.var_name)) == 0

        _ = [self.tolist()] if len(self.shape) == 1 else self.tolist()
        __ = [other.tolist()] if len(other.shape) == 1 else other.tolist()
        return Solution(
            _ + __,
            self.fitness.tolist() + other.fitness.tolist(),
            self.n_eval.tolist() + other.n_eval.tolist(),
            var_name=self.var_name,
            fitness_name=self.fitness_name,
            index=self.index.tolist() + other.index.tolist(),
            n_obj=self.n_obj,
            verbose=self.verbose,
        )

    def __mul__(self, N: int) -> Solution:
        """repeat a solution N times"""
        assert isinstance(N, int)
        if self.N > 1:
            raise Exception("Replication is not supported for 2D")
        return Solution(
            [self.tolist()] * N,
            self.fitness.tolist() * N,
            self.n_eval.tolist() * N,
            var_name=self.var_name,
            fitness_name=self.fitness_name,
            n_obj=self.n_obj,
            verbose=self.verbose,
        )

    def __rmul__(self, N: int) -> Solution:
        return self.__mul__(N)

    def __setattr__(self, name, value):
        attr = getattr(self, name, None)
        if hasattr(attr, "__iter__") and name in ["fitness", "n_eval", "index"]:
            attr[:] = value  # NOTE: copy the value (not reference) to the attribute
        else:
            super(Solution, self).__setattr__(name, value)

    def __str__(self) -> str:
        var_name = self.var_name.tolist()
        headers = var_name + ["n_eval"] + self.fitness_name if self.verbose else var_name
        if len(self.shape) == 1:
            t = (
                [self.tolist() + self.n_eval.tolist() + self.fitness.tolist()]
                if self.verbose
                else [self.tolist()]
            )
        else:
            t = np.c_[self, self.n_eval, self.fitness].tolist() if self.verbose else self.tolist()

        return tabulate(t, headers=headers, showindex=self.index.tolist(), tablefmt="grid")

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, index):
        _, __ = index, slice(None, None)
        if isinstance(index, tuple):
            _ = index[0]
            if len(index) == 2:
                if isinstance(index[1], int) and not isinstance(index[0], int):
                    __ = slice(index[1], index[1] + 1)
                    index = (_, __)
                else:
                    __ = index[1]

        subarr = super().__getitem__(index)

        # sub-slicing the attributes
        if isinstance(subarr, Solution):
            # NOTE: `slice` is needed here to make sure an array is always returned
            # after slicing attributes `fitness`, `n_eval, `index`
            # Otherwise setting the attribute of a slice is by value...
            _ = slice(_, _ + 1) if isinstance(_, (int, np.int_)) else _
            if len(self.shape) == 1:  # `self` is a 1-d array
                subarr.var_name = subarr.var_name[_]
            else:
                # NOTE: 1-d array should have 1-d `fitness`
                fitness = subarr.fitness[_]
                if len(subarr.shape) == 1:
                    fitness = fitness.ravel()

                super(Solution, subarr).__setattr__("fitness", fitness)
                super(Solution, subarr).__setattr__("n_eval", subarr.n_eval[_])
                super(Solution, subarr).__setattr__("index", subarr.index[_])
                super(Solution, subarr).__setattr__("var_name", subarr.var_name[__])

            subarr.N = 1 if len(subarr.shape) == 1 else subarr.shape[0]
            subarr.dim = subarr.shape[0] if len(subarr.shape) == 1 else subarr.shape[1]

        return subarr

    def __array_finalize__(self, obj: Solution):
        """
        `__array_finalize__` is called after new `Solution` instance is created: from calling
        1) `__new__`, 2) view casting (`ndarray`.`view()`) or 3) slicing (`__getitem__`)
        """
        if obj is None:
            return
        # Needed for array slicing (__getitem__)
        super(Solution, self).__setattr__("fitness", getattr(obj, "fitness", None))
        super(Solution, self).__setattr__("n_eval", getattr(obj, "n_eval", None))
        super(Solution, self).__setattr__("index", getattr(obj, "index", None))

        # NOTE: to get rid of pylint errors..
        self.index: np.ndarray = getattr(obj, "index", None)
        self.var_name: np.ndarray = getattr(obj, "var_name", None)
        self.fitness: np.ndarray = getattr(obj, "fitness", None)
        self.fitness_name: List[str] = getattr(obj, "fitness_name", None)
        self.n_eval: np.ndarray = getattr(obj, "n_eval", None)
        self.verbose: bool = getattr(obj, "verbose", None)
        self.dim: int = getattr(obj, "dim", None)
        self.N: int = getattr(obj, "N", None)
        self.n_obj: int = getattr(obj, "n_obj", None)

    @classmethod
    def from_dict(
        cls,
        x: Union[List[dict], dict],
        index: List[int, str] = None,
        var_name: Union[str, List[str]] = None,
        **kwargs,
    ) -> Solution:
        if isinstance(x, dict):
            var_name = list(x.keys())
            res = cls.__new__(cls, x=list(x.values()), var_name=var_name, index=index, **kwargs)
        elif isinstance(x, list):
            var_name = list(x[0].keys())
            _x = [list(_.values()) for _ in x]
            res = cls.__new__(cls, x=_x, var_name=var_name, index=index, **kwargs)
        return res

    def unique(self) -> Solution:
        if len(self.shape) == 1:
            return self
        _, index = np.unique(self.tolist(), axis=0, return_index=True)
        return self[np.sort(index)]

    def to_list(self) -> List:
        return super().tolist()

    def to_dict(self, orient: str = "index", with_index: bool = False) -> Union[List, Dict]:
        """A Solution object to a dictionary

        Parameters
        ----------
        orient : str, optional
            orientation of the dictionary, which is either `index` or `var`, by default "index"
        with_index : bool, optional
            whether indices of solutions should be used in the dictionary, by default False
        """
        if orient not in ("index", "var"):
            raise ValueError(f"`orient` value {orient} is invalid")

        obj = np.atleast_2d(self.view(np.ndarray))
        if orient == "index":
            if with_index:
                res = {
                    _index: {self.var_name[k]: obj[i, k] for k in range(self.dim)}
                    for i, _index in enumerate(self.index)
                }
            else:
                res = [
                    {self.var_name[k]: obj[i, k] for k in range(self.dim)}
                    for i, _index in enumerate(self.index)
                ]
                if len(self.shape) == 1:
                    res = res[0]
        elif orient == "var":
            if with_index:
                res = {
                    _name: {index: obj[i, k] for i, index in enumerate(self.index)}
                    for k, _name in enumerate(self.var_name)
                }
            else:
                res = {_name: list(obj[:, k]) for k, _name in enumerate(self.var_name)}
        return res

    def to_csv(
        self,
        fname: str,
        delimiter: str = ",",
        append: bool = False,
        header: bool = True,
        index: bool = True,
        attribute: bool = True,
    ):
        """Convert to the CSV format

        Parameters
        ----------
        fname : str
            the name of the output CSV file
        delimiter : str, optional
            column-separating delimiter, by default ","
        append : bool, optional
            whether to append to an existing file, by default False
        header : bool, optional
            whether to write a header in the CSV, by default True
        index : bool, optional
            wheter to include the index, by default True
        attribute : bool, optional
            whether to include the atttributes, i.e., `self.n_eval` and `self.fitness`,
            by default True
        """
        var_name = self.var_name.tolist()
        if header:
            _header = var_name
            if index:
                _header = [""] + _header
            if attribute:
                attr_name = ["n_eval"] + self.fitness_name
                _header += attr_name
            _header = delimiter.join(_header) + "\n"

        data = self.reshape(1, -1) if len(self.shape) == 1 else self
        if index:
            data = np.c_[self.index, data]
        if attribute:
            data = np.c_[data, self.n_eval, self.fitness]

        out = [delimiter.join(map(str, row)) + "\n" for row in data.tolist()]
        mode = "a" if append else "w"
        with open(fname, mode) as f:
            if header:
                f.writelines(_header)
            f.writelines(out)
