from __future__ import annotations

import ast
import functools
import itertools
import json
from collections import Counter
from copy import copy, deepcopy
from itertools import chain
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.random import rand
from pyDOE import lhs
from scipy.special import logit
from sobol_seq import i4_sobol_generate

from .._exception import ConstraintEvaluationError
from .node import Node
from .samplers import SCMC
from .variable import Bool, Discrete, Integer, Ordinal, Real, Subset, Variable

__authors__ = "Hao Wang"

_reduce = lambda iterable: functools.reduce(lambda a, b: a + b, iterable)


class SearchSpace:
    """Search Space Base Class

    Attributes
    ----------
    public:
        dim : int,
            dimensinality of the search space
        bounds : a list of lists,
            each sub-list stores the lower and upper bound for continuous/ordinal variable
            and categorical values for nominal variable
        var_name : a list of str,
            variable names per dimension
        var_type : a list of str,
            variable type per dimension, 'Real': continuous/real-valued,
            'Discrete': discrete, 'Integer': ordinal/integer-valued
        r_mask : a bool array,
            the mask array for continuous variables
        i_mask : a bool array,
            the mask array for integer variables
        d_mask : a bool array,
            the mask array for discrete variables
        id_r : an int array,
            the index array for continuous variables
        id_i : an int array,
            the index array for integer variables
        id_d : an int array,
            the index array for discrete variables

    protected:
        _levels : a dict of lists,
            unique levels for each categorical/discrete variable. The index of
            the corresponding discrete variable serves as the dictionary key.
    """

    _supported_types = (Real, Integer, Ordinal, Discrete, Bool)

    def __init__(
        self,
        data: List[Variable],
        random_seed: int = None,
        structure: Union[dict, List[Node]] = None,
    ):
        """Search Space

        Parameters
        ----------
        data : List[Variable]
            a list of variables consistuting the search space
        random_seed : int, optional
            random seed controlling the `sample` function, by default None
        """
        # declarations to fix the pylint warnings..
        self._var_name: List[str] = []
        self._var_type: List[str] = []
        self._bounds: List[tuple] = []
        self._levels: dict = {}

        self.random_seed: int = random_seed
        self._set_data(data)
        self.__set_structure(structure)
        SearchSpace.__set_type(self)

    @property
    def var_name(self):
        return self._var_name
        # return (self._var_name[0] if self.dim == 1 else self._var_name)

    @var_name.setter
    def var_name(self, var_name):
        if isinstance(var_name, str):
            var_name = [var_name + str(_) for _ in range(self.dim)]
        else:
            assert len(var_name) == self.dim

        for i, name in enumerate(var_name):
            self.data[i].name = name
        self._set_data(self.data)

    @property
    def var_type(self):
        return self._var_type

    @property
    def bounds(self):
        return self._bounds

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed):
        if seed:
            seed = int(seed)
        self._random_seed = seed
        np.random.seed(self._random_seed)

    @staticmethod
    def _ready_args(bounds, var_name, **kwargs):
        """infer the dimension, set the variable name, and ready other arguments"""
        if hasattr(bounds[0], "__iter__") and not isinstance(bounds[0], str):
            bounds = [tuple(b) for b in bounds]
        else:
            bounds = [tuple(bounds)]
        dim = len(bounds)
        out: List[Dict] = [{"bounds": bounds[i]} for i in range(dim)]

        if isinstance(var_name, str):
            if dim > 1:
                var_name = [var_name + str(_) for _ in range(dim)]
            else:
                var_name = [var_name]
        assert len(var_name) == dim
        for i in range(dim):
            out[i]["name"] = var_name[i]

        for key, value in kwargs.items():
            if value is not None:
                if not isinstance(value, (tuple, list)):
                    value = [value] * dim
                assert len(value) == dim
                for i in range(dim):
                    out[i][key] = value[i]
        return out

    def _check_data(self, data):
        assert all([isinstance(d, self._supported_types) for d in data])
        names = np.asarray([var.name for var in data])
        for name, count in Counter(names).items():
            if count > 1:
                idx = np.nonzero(names == name)[0]
                _names = [name + str(i) for i in range(count)]
                for i, k in enumerate(idx):
                    data[k].name = _names[i]

    def __set_structure(self, structure: Union[dict, List[Node]] = None):
        if structure is None:
            structure = dict()
            for var in self.data:
                if var.conditions is None:
                    continue
                # TODO: support more dependent variables in the condition
                key = var.conditions["vars"][0]
                if key not in var.conditions["vars"]:
                    raise ValueError(f"variable {var} not in {self}")
                structure.setdefault(key, []).append(
                    {"name": var.name, "condition": var.conditions["string"]}
                )
            self.structure = Node.from_dict(structure)
        # a list of tree/nodes
        elif isinstance(structure, list) and all([isinstance(t, Node) for t in structure]):
            self.structure = [tree.remove(self.var_name, invert=True) for tree in structure]
        elif isinstance(structure, dict):  # dictionary input
            self.structure = [
                tree.remove(self.var_name, invert=True) for tree in Node.from_dict(structure)
            ]

    @staticmethod
    def __set_type(obj: SearchSpace) -> SearchSpace:
        _type = np.unique(obj.var_type)
        obj.__class__ = eval(_type[0] + "Space") if len(_type) == 1 else SearchSpace
        return obj

    def _set_data(self, data):
        """Sanity check on the input data and set the auxiliary variables"""
        self._check_data(data)
        self.data: List = data
        self.dim: int = len(self.data)
        self._bounds = [var.bounds for var in self.data]
        self._var_type = [type(v).__name__ for v in self.data]
        self._var_name = [v.name for v in self.data]
        self._set_index()
        self._set_levels()

    def _set_index(self):
        """set indices for each type of variables"""
        if self.dim > 0:
            for var_type in self._supported_types:
                name = var_type.__name__
                attr_mask = name.lower() + "_mask"
                attr_id = name.lower() + "_id"
                mask = np.asarray(self._var_type) == name
                self.__dict__[attr_mask] = mask
                self.__dict__[attr_id] = np.nonzero(mask)[0]

            self.categorical_id = np.r_[self.discrete_id, self.ordinal_id, self.bool_id]
            self.categorical_mask = np.bitwise_or(
                self.bool_mask, np.bitwise_or(self.discrete_mask, self.ordinal_mask)
            )

    def _set_levels(self):
        # TODO: check if this is still needed
        """Set categorical levels for all nominal variables"""
        if self.dim > 0:
            self.levels = (
                {i: self._bounds[i] for i in self.categorical_id}
                if len(self.categorical_id) > 0
                else {}
            )

    def __getitem__(self, index) -> Union[SearchSpace, Variable]:
        if isinstance(index, (int, slice)):
            data = self.data[index]
        elif hasattr(index, "__iter__") and not isinstance(index, str):
            index = np.array(index)
            if index.dtype.type is np.str_:  # list of variable names
                index = [np.nonzero(np.array(self.var_name) == i)[0][0] for i in index]
            elif index.dtype == bool:  # mask array
                index = np.nonzero(index)[0]
            data = [self.data[i] for i in index]
        elif isinstance(index, str):  # slicing one variable by name
            index = np.nonzero(np.array(self.var_name) == index)[0][0]
            data = self.data[index]
        else:
            raise Exception(f"index type {type(index)} is not supported")

        if isinstance(data, Variable):
            out = data
        elif isinstance(data, list):
            out = SearchSpace(data, self.random_seed)
            # getattr(out, "__set_structure")(self.structure)
        return out

    def __setitem__(self, index, value):
        if isinstance(index, (int, slice)):
            self.data[index] = value
        elif isinstance(index, str):
            index = np.nonzero(np.array(self.var_name) == index)[0][0]
            self.data[index] = value
        elif hasattr(index, "__iter__") and not isinstance(index, str):
            if not hasattr(value, "__iter__") or isinstance(value, str):
                value = [value] * len(index)
            for i, v in zip(index, value):
                if isinstance(i, str):
                    k = np.nonzero(np.array(self.var_name) == i)[0][0]
                    self.data[k] = v
                elif isinstance(i, int):
                    self.data[i] = v
                else:
                    raise Exception(f"index type {type(i)} is not supported")
        self._set_data(self.data)

    def __contains__(self, item: Union[str, Variable, SearchSpace, list, dict]) -> bool:
        """check if a name, a variable, a space, or a sample point in the the search space"""
        if isinstance(item, str):
            return item in self.var_name
        if isinstance(item, Variable):
            return item in self.data
        if isinstance(item, SearchSpace):
            return all(map(lambda x: x in self.data, item.data))
        if isinstance(item, list):
            return all([v in self.__getitem__(i) for i, v in enumerate(item)])
        if isinstance(item, dict):
            return all([v in self.__getitem__(i) for i, v in item.items()])
        raise ValueError(f"type {type(item)} is not supported")

    def __len__(self):
        return self.dim

    def __iter__(self) -> Variable:
        i = 0
        while i < self.dim:
            yield self.__getitem__(i)
            i += 1

    def __eq__(self, cs: SearchSpace) -> bool:
        return self.dim == cs.dim and set(self.data) == set(cs.data)

    def __ne__(self, cs: SearchSpace) -> bool:
        return not self.__eq__(cs)

    def __add__(self, space) -> SearchSpace:
        """Direct Sum of two `SearchSpace` instances"""
        assert isinstance(space, SearchSpace)
        # NOTE: the random seed of `self` has the priority
        random_seed = self.random_seed if self.random_seed else space.random_seed
        data = deepcopy(self.data) + space.data
        structure = [t.deepcopy() for t in self.structure] + [
            t.deepcopy() for t in space.structure
        ]
        return SearchSpace(data, random_seed, structure)

    def __radd__(self, space) -> SearchSpace:
        return self.__add__(space)

    def __iadd__(self, space) -> SearchSpace:
        assert isinstance(space, SearchSpace)
        self._set_data(self.data + space.data)
        SearchSpace.__set_type(self)
        return self

    def __sub__(self, space) -> SearchSpace:
        """Substraction of two `SearchSpace` instances"""
        assert isinstance(space, SearchSpace)
        random_seed = self.random_seed if self.random_seed else space.random_seed
        _res = set(self.var_name) - set(space.var_name)
        _index = [self.var_name.index(_) for _ in _res]
        data = [copy(self.data[i]) for i in range(self.dim) if i in _index]
        cs = SearchSpace(data, random_seed)
        # getattr(cs, "__set_structure")(self.structure)
        return cs

    def __rsub__(self, space) -> SearchSpace:
        return self.__sub__(space)

    def __isub__(self, space) -> SearchSpace:
        assert isinstance(space, SearchSpace)
        _res = set(self.var_name) - set(space.var_name)
        _index = [self.var_name.index(_) for _ in _res]
        self._set_data([self.data[i] for i in range(self.dim) if i in _index])
        SearchSpace.__set_type(self)
        return self

    def __mul__(self, N: int) -> SearchSpace:
        """Replicate `self` by copy

        Parameters
        ----------
        N : int
            Replicate `self` N times as a copt

        Returns
        -------
        SearchSpace
            a copy of replicated `self`
        """
        data = [deepcopy(var) for _ in range(max(1, int(N))) for var in self.data]
        # TODO: this is not working yet..
        structure = [t.deepcopy() for _ in range(max(1, int(N))) for t in self.structure]
        obj = SearchSpace(data, self.random_seed, structure)
        obj.__class__ = type(self)
        return obj

    def __rmul__(self, N: int) -> SearchSpace:
        return self.__mul__(N)

    def __imul__(self, N: int) -> SearchSpace:
        """Incrementally replicate

        Parameters
        ----------
        N : int
            Replicate `self` N times

        Returns
        -------
        SearchSpace
            `self`
        """
        self._set_data(
            self.data + [deepcopy(var) for _ in range(max(1, int(N - 1))) for var in self.data]
        )
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        msg = f"{type(self).__name__} of {self.dim} variables: \n"
        for var in self.data:
            msg += str(var) + "\n"
        return msg

    def pprint(self):
        if self.structure:
            for root in self.structure:
                root.pprint(data={k: self[k] for k in self.var_name})
        else:
            print(self.__str__())

    def filter(self, keys: List[str], invert=False) -> SearchSpace:
        """filter a search space based on a list of variable names

        Parameters
        ----------
        keys : List[str]
            the list of variable names to keep

        Returns
        -------
        Union[Variable, SearchSpace]
            the resulting subspace
        """
        masks = [v in keys for v in self.var_name]
        if invert:
            masks = np.bitwise_not(masks)
        return self[masks]

    @classmethod
    def concat(cls, *args: Tuple[SearchSpace]):
        if len(args) == 1:
            return args[0]

        assert isinstance(args[0], SearchSpace)
        data = list(chain.from_iterable([deepcopy(cs.data) for cs in args]))
        structure = [t.deepcopy() for cs in args for t in cs.structure]
        return SearchSpace(data, structure=structure)

    def pop(self, index: int = -1) -> Variable:
        value = self.data.pop(index)
        self._set_data(self.data)
        self.__set_structure(self.structure)
        SearchSpace.__set_type(self)
        return value

    def remove(self, index: Union[int, str]) -> SearchSpace:
        if isinstance(index, str):
            _index = np.nonzero(np.array(self._var_name) == index)[0]
            if len(_index) == 0:
                raise KeyError(f"The input key {index} not found in `var_name`!")
            _index = _index[0]
        elif hasattr(index, "__iter__"):
            raise KeyError("Multiple indices are not allowed!")
        else:
            _index = index

        self.data.pop(_index)
        self._set_data(self.data)
        self.__set_structure(self.structure)
        return SearchSpace.__set_type(self)

    def update(self, space: SearchSpace) -> SearchSpace:
        """Update the search space based on the var_name of the input search space,
        which behaves similarly to the dictionary update. Please note its difference
        to ``self.__add__``. This function will not update the structure of the search space.

        Parameters
        ----------
        space : SearchSpace
            the input space for the update
        """
        _var_name = space.var_name
        _update = np.array(list(set(_var_name) & set(self.var_name)))
        _add = np.array(list(set(_var_name) - set(self.var_name)))

        _index_update_to = np.nonzero(np.array(self._var_name) == _update)[0]
        _index_update_from = np.nonzero(np.array(_var_name) == _update)[0]
        for i, k in enumerate(_index_update_to):
            j = _index_update_from[i]
            self.data[k] = deepcopy(space.data[j])

        _index_add = np.nonzero(np.array(_var_name) == _add)[0]
        self.data += [space.data[i] for i in _index_add]
        self._set_data(self.data)
        return SearchSpace.__set_type(self)

    def sample(
        self,
        N: int = 1,
        method: str = "uniform",
        h: Callable = None,
        g: Callable = None,
        tol: float = 1e-2,
    ) -> np.ndarray:
        """Sample random points from the search space

        Parameters
        ----------
        N : int, optional
            the number of points to generate, by default 1
        method : str, optional
            the sampling strategy, by default 'uniform'
        h : Callable, optional
            equality constraints, by default None
        g : Callable, optional
            inequality constraints, by default None
        tol : float, optional
            the tolerance on the constraint

        NOTES
        -----
        At this moment, the constraints are handled using the simple Monte Carlo sampling

        Returns
        -------
        np.ndarray
            the sample points in shape `(N, self.dim)`
        """
        # 10 is the minimal number of sample points to take under constraints
        n = max(N, 10) if h or g else N
        constraints = lambda x: np.r_[np.abs(h(x)) if h else [], np.array(g(x)) if g else []]
        S = SCMC(self, constraints, tol=tol).sample(n) if h or g else self._sample(N, method)
        try:
            # NOTE: equality constraints are converted to an epsilon-tude around the
            # corresponding manifold
            idx_h = (
                list(map(lambda x: all(np.isclose(np.abs(h(x)), 0, atol=tol)), S))
                if h
                else [True] * n
            )
            idx_g = list(map(lambda x: np.all(np.asarray(g(x)) <= 0), S)) if g else [True] * n
            idx = np.bitwise_and(idx_h, idx_g)
            S = S[idx, :]
        except Exception as e:
            raise ConstraintEvaluationError(S, str(e)) from None

        # get unique rows
        # S = np.array([list(x) for x in set(tuple(x) for x in S)], dtype=object)
        if len(S) > N:
            S = S[np.random.choice(len(S), N, replace=False)]
        return S

    def _sample(self, N: int = 1, method: str = "uniform") -> np.ndarray:
        # in case this space is empty after slicing
        if self.dim == 0:
            return np.empty(0)

        N = max(int(N), 1)
        X = np.empty((N, self.dim), dtype=object)
        for var_type in self._supported_types:
            attr_id = var_type.__name__.lower() + "_id"
            index = self.__dict__[attr_id]
            if len(index) > 0:  # if such a type of variables exist.
                X[:, index] = getattr(self[index], "_sample")(N, method)
        return X

    def round(self, X: Union[np.ndarray, List[List]]) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=object)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if len(self.real_id) > 0:  # if real-valued variables exist.
            r_subspace = self.__getitem__(self.real_id)
            X[:, self.real_id] = r_subspace.round(X[:, self.real_id].astype(float))
        return X

    def to_linear_scale(self, X: Union[np.ndarray, List[List]]) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=object)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if len(self.real_id) > 0:  # if real-valued variables exist.
            r_subspace = self.__getitem__(self.real_id)
            X[:, self.real_id] = r_subspace.to_linear_scale(X[:, self.real_id].astype(float))
        return X

    def to_dict(self) -> dict:
        out: dict = {}
        for _, var in enumerate(self.data):
            value = {
                "range": var.bounds,
                "type": type(var),
                "N": 1,
            }
            if isinstance(var, Real):
                value["precision"] = var.precision
                value["scale"] = var.scale
            elif isinstance(var, Integer):
                value["step"] = var.step

            if hasattr(var, "conditions"):
                value["conditions"] = var.conditions
                # TODO: also export var._action?
            out[var.name] = value
        return out

    def to_json(self, file: str):
        with open(file, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, param: dict) -> SearchSpace:
        """Create a search space object from input dictionary

        Parameters
        ----------
        param : dict
            A dictionary that describes the search space
        space_name : bool, optional
            Whether a (multi-dimensional) subspace should be named. If this named space
            is a subspace a whole search space, for a solution sampled from the whole space, its
            components pertaining to this subspace will be grouped together under the key
            `space_name`, when this solution is converted to a dictionary/json
            (see `SearchSpace.to_dict`).
        source : string, optional
            Where the dictionary originates from. Can be either 'default' or 'irace'

        Returns
        -------
        SearchSpace
        """
        # TODO: implement `grouping`
        assert isinstance(param, dict)

        variables = []
        for k, v in param.items():
            if "range" in v:
                bounds = v["range"]
                if not hasattr(bounds[0], "__iter__") or isinstance(bounds[0], str):
                    bounds = tuple(bounds)
            else:
                bounds = ()

            N = range(int(v["N"])) if "N" in v else range(1)
            default_value = v["defualt"] if "default" in v else None
            if v["type"] in ["r", "real"]:  # real-valued parameter
                precision = v["precision"] if "precision" in v else None
                scale = v["scale"] if "scale" in v else "linear"
                _vars = [
                    Real(
                        bounds,
                        name=k,
                        default_value=default_value,
                        precision=precision,
                        scale=scale,
                    )
                    for _ in N
                ]
            elif v["type"] in ["i", "int", "integer"]:  # integer-valued parameter
                _vars = [
                    Integer(bounds, name=k, default_value=default_value, step=v.pop("step", 1))
                    for _ in N
                ]
            elif v["type"] in ["o", "ordinal"]:  # ordinal parameter
                _vars = [Ordinal(bounds, name=k, default_value=default_value) for _ in N]
            elif v["type"] in ["c", "cat"]:  # category-valued parameter
                _vars = [Discrete(bounds, name=k, default_value=default_value) for _ in N]
            elif v["type"] in ["s", "subset"]:  # subset parameter
                _vars = [Subset(bounds, name=k, default_value=default_value) for _ in N]
            elif v["type"] in ["b", "bool"]:  # Boolean-valued
                _vars = [Bool(name=k, default_value=default_value) for _ in N]
            variables += _vars
        return SearchSpace(variables)

    @classmethod
    def from_json(cls, file: str) -> SearchSpace:
        """Create a seach space from a json file

        Parameters
        ----------
        file : str
            Path to the input json file

        Returns
        -------
        SearchSpace
            an `SearchSpace` object converted from the json file
        """
        with open(file, "r") as f:
            return cls.from_dict(json.load(f))

    def get_unconditional_subspace(self) -> List[Tuple[dict, SearchSpace]]:
        """get all unconditional subspaces"""
        if self.structure:
            # all variables in the conditional structure
            _var = _reduce([t.get_all_name() for t in self.structure])
            # remaining variables not affected by conditions
            isolated_var = [self[v] for v in set(self.var_name) - set(_var)]
            out, d = list(), dict()
            # all paths in the conditional tree
            paths = [list(root.get_all_path().items()) for root in self.structure]
            idx = [list(range(len(_))) for _ in paths]
            # get all combinations of paths from all trees
            for item in itertools.product(*idx):
                condition = _reduce([paths[i][k][0] for i, k in enumerate(item)])
                variables = _reduce([paths[i][k][1] for i, k in enumerate(item)])
                d[condition] = variables
            # create all unconditional subspaces
            for condition, var in d.items():
                key = {
                    t.body[0].value.left.id: t.body[0].value.comparators[0].value
                    for t in map(ast.parse, condition)
                }
                out.append((key, SearchSpace(isolated_var + [self[v] for v in var])))
            # TODO: consider the case where selector/conditioning variable has other values
        else:
            out = [({}, self)]
        return out


class RealSpace(SearchSpace):
    """Space of real values"""

    def __init__(
        self,
        bounds: List,
        var_name: Union[str, List[str]] = "real",
        default_value: Union[float, List[float]] = None,
        precision: Union[int, List[int]] = None,
        scale: Union[str, List[str]] = None,
        **kwargs,
    ):
        out = self._ready_args(
            bounds, var_name, default_value=default_value, precision=precision, scale=scale
        )
        data = [Real(**_) for _ in out]
        super().__init__(data, **kwargs)

    def _sample(self, N: int = 1, method: str = "uniform") -> np.ndarray:
        bounds = np.array([var._bounds_transformed for var in self.data])
        lb, ub = bounds[:, 0], bounds[:, 1]
        if method == "uniform":  # uniform random samples
            X = (ub - lb) * rand(N, self.dim) + lb
        elif method == "LHS":  # Latin hypercube sampling
            if N == 1:
                X = (ub - lb) * rand(N, self.dim) + lb
            else:
                X = (ub - lb) * lhs(self.dim, samples=N, criterion="maximin") + lb
        elif method == "sobol":
            X = (ub - lb) * i4_sobol_generate(self.dim, N) + lb
        return self.round(self.to_linear_scale(X))

    def round(self, X: Union[np.ndarray, List[List]]) -> np.ndarray:
        X = np.atleast_2d(X).astype(float)
        assert X.shape[1] == self.dim
        for i, var in enumerate(self.data):
            X[:, i] = var.round(X[:, i])
        return X

    def to_linear_scale(self, X: Union[np.ndarray, List[List]]) -> np.ndarray:
        X = np.atleast_2d(X).astype(float)
        assert X.shape[1] == self.dim
        for i, var in enumerate(self.data):
            X[:, i] = var.to_linear_scale(X[:, i])
        return X


class _DiscreteSpace(SearchSpace):
    """Space of discrete values"""

    def round(self, X: Union[np.ndarray, List[List]]) -> np.ndarray:
        """do nothing since this method is not valid for this class"""
        return X

    def to_linear_scale(self, X: Union[np.ndarray, List[List]]) -> np.ndarray:
        """do nothing since this method is not valid for this class"""
        return X

    def _sample(self, N: int = 1, method: str = "uniform") -> np.ndarray:
        if isinstance(self, IntegerSpace):
            dtype = int
        elif isinstance(self, BoolSpace):
            dtype = bool
        else:
            dtype = object

        X = np.empty((N, self.dim), dtype=dtype)
        for i in range(self.dim):
            X[:, i] = self.data[i].sample(N, method=method)
        return X


class SubsetSpace(_DiscreteSpace):
    """A discrete space created by enumerating all subsets of the input `bounds`"""

    def __init__(
        self,
        bounds: List,
        var_name: Union[str, List[str]] = "subset",
        default_value: Union[int, List[int]] = None,
        **kwargs,
    ):
        out = self._ready_args(bounds, var_name, default_value=default_value)
        data = [Subset(**_) for _ in out]
        super().__init__(data, **kwargs)


class IntegerSpace(_DiscreteSpace):
    """Space of contiguous integer values"""

    def __init__(
        self,
        bounds: List,
        var_name: Union[str, List[str]] = "integer",
        default_value: Union[int, List[int]] = None,
        step: Optional[Union[int, float]] = 1,
        **kwargs,
    ):
        out = self._ready_args(bounds, var_name, default_value=default_value, step=step)
        data = [Integer(**_) for _ in out]
        super().__init__(data, **kwargs)


class OrdinalSpace(_DiscreteSpace):
    """Space of ordinal values"""

    def __init__(
        self,
        bounds: List[str, int, float],
        var_name: Union[str, List[str]] = "ordinal",
        default_value=None,
        **kwargs,
    ):
        out = self._ready_args(bounds, var_name, default_value=default_value)
        data = [Ordinal(**_) for _ in out]
        super().__init__(data, **kwargs)


class DiscreteSpace(_DiscreteSpace):
    """Space of discrete values"""

    def __init__(
        self,
        bounds: List[str, int, float],
        var_name: Union[str, List[str]] = "discrete",
        default_value=None,
        **kwargs,
    ):
        out = self._ready_args(bounds, var_name, default_value=default_value)
        data = [Discrete(**_) for _ in out]
        super().__init__(data, **kwargs)


class BoolSpace(_DiscreteSpace):
    """Space of Bool values"""

    def __init__(
        self,
        var_name: Union[str, List[str]] = "bool",
        default_value: Union[bool, List[bool]] = None,
        **kwargs,
    ):
        out = self._ready_args((False, True), var_name, default_value=default_value)
        data = [Bool(**_) for _ in out]
        super().__init__(data, **kwargs)
