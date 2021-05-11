from __future__ import annotations

import json
import re
from abc import ABC
from collections import Counter
from copy import copy, deepcopy
from itertools import chain
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.random import rand, randint
from pyDOE import lhs
from scipy.special import logit

__authors__ = "Hao Wang"

TRANS = {
    "linear": [lambda x: x, [-np.inf, np.inf]],
    "log": [np.log, [0, np.inf]],
    "log10": [np.log10, [0, np.inf]],
    "logit": [logit, [0, 1]],
    "bilog": [lambda x: np.sign(x) * np.log(1 + np.abs(x)), [-np.inf, np.inf]],
}
INV_TRANS = {
    "linear": lambda x: x,
    "log": np.exp,
    "log10": lambda x: np.power(10, x),
    "logit": lambda x: 1 / (1 + np.exp(-x)),
    "bilog": lambda x: np.sign(x) * (np.exp(np.abs(x)) - 1),
}

# TODO: discuss and fix the return value of `sample`, `round`, and `to_linear_scale`


class Variable(ABC):
    """Base class for decision variables"""

    def __init__(
        self,
        bounds: List[int, float, str],
        name: str,
        default_value: Union[int, float, str] = None,
        conditions: str = None,
        action: Union[callable, int, float, str] = lambda x: x,
    ):
        """Base class for decision variables

        Parameters
        ----------
        bounds : List[int, float, str]
            a list/tuple giving the range of the variable.
                * For `Real`, `Integer`: (lower, upper)
                * For `Ordinal` and `Discrete`: (value1, value2, ...)
        name : str
            variable name
        default_value : Union[int, float, str], optional
            default value, by default None
        conditions : str, optional
            a string specifying the condition on which the variable is problematic, e.g.,
            being either invalid or ineffective, by default None. The variable name in
            this string should be quoted as `var name`. Also, you could use multiple
            variables and logic conjunctions/disjunctions therein.
            Example: "`var1` == True and `var2` == 2"
        action : Union[callable, int, float, str], optional
            the action to take when `condition` evaluates to True, by default `lambda x: x`.
            It can be simply a fixed value to which the variable will be set, or a callable
            that determines which value to take.
        """
        if len(bounds) > 0 and isinstance(bounds[0], list):
            bounds = bounds[0]
        self.name: str = name
        self.bounds = tuple(bounds)
        self.default_value = default_value
        self.add_conditions(conditions, action)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        msg = f"{self.name} -> {type(self).__name__} | range: {self.bounds}"
        if self.default_value is not None:
            msg += f" | default: {self.default_value}"
        return msg

    def __eq__(self, var: Variable) -> bool:
        return (
            self.__class__ == type(var)
            and self.bounds == var.bounds
            and self.default_value == var.default_value
            and self.name == var.name
            and self._conditions == var._conditions
        )  # TODO: verify this!

    def __ne__(self, var: Variable) -> bool:
        return not self.__eq__(var)

    def add_conditions(self, conditions: str, action: Union[callable, int, float, str]):
        self._conditions = None
        if conditions is not None:
            self.var_in_conditions = re.findall(r"`([^`]*)`", conditions)
            for i, var_name in enumerate(self.var_in_conditions):
                conditions = conditions.replace(f"`{var_name}`", f"#{i}")
            self._conditions = conditions

        if isinstance(action, (int, float, str)):
            self._action = lambda x: action
        elif hasattr(action, "__call__"):
            self._action = action

    @property
    def conditions(self):
        if self._conditions is None:
            return None
        out = copy(self._conditions)
        for i, var_name in enumerate(self.var_in_conditions):
            out = out.replace(f"#{i}", f"{var_name}")
        return out

    @property
    def action(self):
        return self._action


class Real(Variable):
    """Real-valued variable taking its value in a continuum"""

    def __init__(
        self,
        bounds: Tuple[float, float],
        name: str = "r",
        default_value: float = None,
        precision: int = None,
        scale: str = "linear",
        **kwargs,
    ):
        """Real-valued variable taking its value in a continuum

        Parameters
        ----------
        bounds : [Tuple[float, float]
            the lower and upper bound
        name : str, optional
            the variable name, by default 'r'
        default_value : float, optional
            the default value, by default None
        precision : int, optional
            the number of digits after decimal, by default None
        scale : str, optional
            the scale on which uniform sampling is performed, by default 'linear'
        """
        assert bounds[0] < bounds[1]
        assert scale in TRANS.keys()
        assert precision is None or isinstance(precision, int)
        super().__init__(bounds, name, default_value, **kwargs)
        self.precision: int = precision
        self.scale = scale

    def __hash__(self):
        return hash((self.name, self.bounds, self.default_value, self.precision, self.scale))

    def __str__(self):
        msg = super().__str__()
        if self.precision:
            msg += f" | precision: .{self.precision}f"
        msg += f" | scale: {self.scale}"
        return msg

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        if scale is None:
            scale = "linear"

        assert scale in TRANS.keys()
        self._scale: str = scale
        self._trans: Callable = TRANS[scale][0]
        self._inv_trans: Callable = INV_TRANS[scale]
        _range = TRANS[scale][1]

        if (self.bounds[0] < _range[0]) or (self.bounds[0] > _range[1]):
            raise ValueError(
                f"lower bound {self.bounds[0]} not in the working "
                f"range of the given scale {self._scale}"
            )

        if (self.bounds[1] < _range[0]) or (self.bounds[1] > _range[1]):
            raise ValueError(
                f"upper bound {self.bounds[1]} not in the working"
                f"of the given scale {self._scale}"
            )
        self._bounds_transformed = self._trans(self.bounds)

    def to_linear_scale(self, X):
        return X if self.scale == "linear" else self._inv_trans(X)

    def round(self, X):
        """Round the real-valued components of `X` to the
        corresponding numerical precision, if given
        """
        X = deepcopy(X)
        if self.precision is not None:
            X = np.round(X, self.precision)
        return X


class _Discrete(Variable):
    """Represents Integer, Ordinal, Bool, and Discrete"""

    def __init__(self, bounds, *args, **kwargs):
        bounds = list(dict.fromkeys(bounds))  # get rid of duplicated levelss
        # map discrete values (bounds) to integers for sampling
        self._map_func: callable = None
        self._size: int = None
        super().__init__(bounds, *args, **kwargs)

    def __hash__(self):
        return hash((self.name, self.bounds, self.default_value))

    def sample(
        self, N: int = 1, method: str = "uniform", h: Callable = None, g: Callable = None
    ) -> List:
        # TODO: to handle `h` and `g`...
        # TODO: `method` is not take into account for now..
        return list(map(self._map_func, randint(0, self._size, N)))


class Discrete(_Discrete):
    """Discrete variable, whose values should come with a linear order"""

    def __init__(self, bounds, name: str = "d", default_value: Union[int, str] = None, **kwargs):
        super().__init__(bounds, name, default_value, **kwargs)
        self._map_func = lambda i: self.bounds[i]
        self._size = len(self.bounds)


# TODO: `bounds` -> `range_`?
class Ordinal(_Discrete):
    """A generic ordinal variable, whose values should come with a linear order"""

    def __init__(self, bounds, name: str = "ordinal", default_value: int = None, **kwargs):
        super().__init__(bounds, name, default_value, **kwargs)
        self._map_func = lambda i: self.bounds[i]
        self._size = len(self.bounds)


class Integer(_Discrete):
    """Integer variable, whose values are contiguous"""

    def __init__(
        self, bounds, name: str = "i", default_value: int = None, step: Optional[int, float] = 1
    ):
        super().__init__(bounds, name, default_value)
        assert len(self.bounds) == 2
        assert self.bounds[0] < self.bounds[1]
        assert all(map(lambda x: isinstance(x, (int, float)), self.bounds))
        self.step = step
        self._map_func = lambda i: self.bounds[0] + i * self.step
        self._size = int(np.floor((self.bounds[1] - self.bounds[0]) / self.step) + 1)

    def __hash__(self):
        return hash((self.name, self.bounds, self.default_value, self.step))

    def __str__(self):
        msg = super().__str__()
        msg += f" | step: {self.step}"
        return msg


class Bool(_Discrete):
    """Boolean variable"""

    def __init__(self, name: str = "bool", default_value: int = True, **kwargs):
        # NOTE: remove `bounds` if it presents in the input
        kwargs.pop("bounds", None)
        assert default_value is None or isinstance(default_value, bool)
        super().__init__((False, True), name, default_value, **kwargs)
        self._map_func = bool
        self._size = 2


class SearchSpace(object):
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

    def __init__(self, data: List[Variable], random_seed: int = None):
        """
        Parameters
        ----------
        data : List
            It should be a list of instances of class `Variable`
        random_seed : int, optional
            The random seed controlling the `sample` function, by default None
        """
        # declarations to fix the pylint warnings..
        self._var_name: List[str] = []
        self._var_type: List[str] = []
        self._bounds: List[tuple] = []
        self._levels: dict = {}

        self.random_seed: int = random_seed
        self._set_data(data)
        SearchSpace.__set_type(self)
        self.__set_conditions()

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

    @staticmethod
    def __set_type(obj: SearchSpace) -> SearchSpace:
        _type = np.unique(obj.var_type)
        if len(_type) == 1:
            obj.__class__ = eval(_type[0] + "Space")
        else:
            obj.__class__ = SearchSpace
        return obj

    def __set_conditions(self):
        # TODO: perhaps implement a `conditions` property (list of all conditions)
        # TODO: to validate the conditions specified in variables
        # TODO: add conditions when the prefix of some variables appears (conditional parameters)
        for var in self.data:
            if var.conditions is not None:
                pre, _, __ = var.name.rpartition(".")
                if pre:
                    for i, var_name in enumerate(var.var_in_conditions):
                        if not var_name.startswith(pre):
                            var.var_in_conditions[i] = pre + "." + var_name

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
            if len(self.categorical_id) > 0:
                self.levels = {i: self._bounds[i] for i in self.categorical_id}
            else:
                self.levels = None

    def __getitem__(self, index) -> SearchSpace:
        if isinstance(index, slice):
            data = self.data[index]
            if not isinstance(data, list):
                data = [data]
        elif isinstance(index, (list, np.ndarray)):
            data = [self.data[index[0]]] if len(index) == 1 else [self.data[i] for i in index]
        else:
            data = [self.data[index]]
        return SearchSpace(data, self.random_seed)

    def __setitem__(self, index, value):
        if isinstance(index, (int, slice)):
            self.data[index] = value
        elif isinstance(index, list):
            for i, v in zip(index, value):
                self.data[i] = v
        self._set_data(self.data)

    def __contains__(self, item: Union[str, Variable, SearchSpace]) -> bool:
        if isinstance(item, str):
            return item in self.var_name
        if isinstance(item, Variable):
            return item in self.data
        if isinstance(item, SearchSpace):
            return all(map(lambda x: x in self.data, item.data))

    def __len__(self):
        return self.dim

    def __iter__(self):
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
        return SearchSpace(data, random_seed)

    def __radd__(self, space) -> SearchSpace:
        return self.__add__(space)

    def __iadd__(self, space) -> SearchSpace:
        assert isinstance(space, SearchSpace)
        self.data += space.data
        self._set_data(self.data)
        SearchSpace.__set_type(self)
        return self

    def __sub__(self, space) -> SearchSpace:
        """Substraction of two `SearchSpace` instances"""
        assert isinstance(space, SearchSpace)
        random_seed = self.random_seed if self.random_seed else space.random_seed
        _res = set(self.var_name) - set(space.var_name)
        _index = [self.var_name.index(_) for _ in _res]
        data = [copy(self.data[i]) for i in range(self.dim) if i in _index]
        return SearchSpace(data, random_seed)

    def __rsub__(self, space) -> SearchSpace:
        return self.__sub__(space)

    def __isub__(self, space) -> SearchSpace:
        assert isinstance(space, SearchSpace)
        _res = set(self.var_name) - set(space.var_name)
        _index = [self.var_name.index(_) for _ in _res]
        self.data = [self.data[i] for i in range(self.dim) if i in _index]
        self._set_data(self.data)
        SearchSpace.__set_type(self)
        return self

    def __mul__(self, N) -> SearchSpace:
        """Replicate a `SearchSpace` N times"""
        data = [deepcopy(var) for _ in range(max(1, int(N))) for var in self.data]
        obj = SearchSpace(data, self.random_seed)
        obj.__class__ = type(self)
        return obj

    def __rmul__(self, N) -> SearchSpace:
        return self.__mul__(N)

    def __imul__(self, N) -> SearchSpace:
        self._set_data(deepcopy(self.data) * max(1, int(N)))
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        msg = f"{type(self).__name__} of {self.dim} variables: \n"
        for var in self.data:
            msg += str(var) + "\n"
        return msg

    @classmethod
    def concat(cls, *args: Tuple[SearchSpace]):
        if len(args) == 1:
            return args[0]

        assert isinstance(args[0], SearchSpace)
        data = list(chain.from_iterable([deepcopy(_.data) for _ in args]))
        return SearchSpace(data)

    def pop(self, index: int = -1) -> Variable:
        value = self.data.pop(index)
        self._set_data(self.data)
        SearchSpace.__set_type(self)
        self.__set_conditions()
        return value

    def remove(self, index: Union[int, str]) -> SearchSpace:
        if isinstance(index, str):
            _index = np.nonzero(np.array(self._var_name) == index)[0]
            if len(_index) == 0:
                raise KeyError(f"The input key {index} not found in `var_name`!")
            else:
                _index = _index[0]
        elif hasattr(index, "__iter__"):
            raise KeyError("Multiple indices are not allowed!")
        else:
            _index = index

        self.data.pop(_index)
        self._set_data(self.data)
        self.__set_conditions()
        return SearchSpace.__set_type(self)

    def update(self, space: SearchSpace) -> SearchSpace:
        """Update the search space based on the var_name of the input search space,
        which behaves similarly to the dictionary update. Please note its difference
        to ``self.__add__``

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
        self, N: int = 1, method: str = "uniform", h: Callable = None, g: Callable = None
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

        Returns
        -------
        np.ndarray
            the sample points in shape `(N, self.dim)`
        """
        # in case this space is empty after slicing
        if self.dim == 0:
            return np.empty(0)

        N = max(int(N), 1)
        X = np.empty((N, self.dim), dtype=object)
        for var_type in self._supported_types:
            attr_id = var_type.__name__.lower() + "_id"
            index = self.__dict__[attr_id]
            if len(index) > 0:  # if such a type of variables exist.
                X[:, index] = self.__getitem__(index).sample(N, method, h=h, g=g)
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
    def from_dict(cls, param: dict, source="default", **kwargs) -> SearchSpace:
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
        if source == "irace":
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
                elif v["type"] in ["b", "bool"]:  # Boolean-valued
                    _vars = [Bool(name=k, default_value=default_value) for _ in N]
                variables += _vars
        elif source == "irace": # the configuration space from irace
            param_names = param["names"]
            cont_params = [x for (x, y) in zip(param_names, param["types"]) if y == "r"]
            ordinal_params = [x for (x, y) in zip(param_names, param["types"]) if y == "i"]
            nominal_params = [
                x for (x, y) in zip(param_names, param["types"]) if y == "c" or y == "o"
            ]
            if len(cont_params) > 0:
                variables += [Real(param["domain"][par], name=par) for par in cont_params]
            if len(ordinal_params) > 0:
                variables += [Ordinal(param["domain"][par], name=par) for par in ordinal_params]
            if len(nominal_params) > 0:
                variables += [Discrete(param["domain"][par], name=par) for par in nominal_params]
        else:
            raise ValueError("This source is not currently supported")
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


# TODO: those classes might not be necessary.. we could move the `sample` method to
# the corresponding variables
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

    def sample(
        self, N: int = 1, method: str = "uniform", h: Callable = None, g: Callable = None
    ) -> np.ndarray:
        # TODO: to handle `h` and `g`
        bounds = np.array([var._bounds_transformed for var in self.data])
        lb, ub = bounds[:, 0], bounds[:, 1]

        if method == "uniform":  # uniform random samples
            X = (ub - lb) * rand(N, self.dim) + lb
        elif method == "LHS":  # Latin hypercube sampling
            if N == 1:
                X = (ub - lb) * rand(N, self.dim) + lb
            else:
                X = (ub - lb) * lhs(self.dim, samples=N, criterion="maximin") + lb
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

    def sample(
        self, N: int = 1, method: str = "uniform", h: Callable = None, g: Callable = None
    ) -> np.ndarray:
        if isinstance(self, IntegerSpace):
            dtype = int
        elif isinstance(self, BoolSpace):
            dtype = bool
        else:
            dtype = object

        X = np.empty((N, self.dim), dtype=dtype)
        for i in range(self.dim):
            X[:, i] = self.data[i].sample(N, method=method, h=h, g=g)
        return X


class IntegerSpace(_DiscreteSpace):
    """Space of contiguous integer values"""

    def __init__(
        self,
        bounds: List,
        var_name: Union[str, List[str]] = "integer",
        default_value: Union[int, List[int]] = None,
        step: Optional[int, float] = 1,
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
