from __future__ import annotations

from typing import List, Dict, Union, Callable, Tuple
from abc import ABC

import json
import functools
from copy import copy, deepcopy
from collections import Counter
from itertools import chain

import numpy as np
from numpy.random import randint, rand

from pyDOE import lhs
from scipy.special import logit
from .solution import Solution

__authors__ = ['Hao Wang']

# TODO: add conditional parameters
TRANS = {
    'linear': [lambda x: x, [-np.inf, np.inf]],
    'log': [np.log, [0, np.inf]],
    'log10': [np.log10, [0, np.inf]],
    'logit': [logit, [0, 1]],
    'bilog': [
        lambda x: np.sign(x) * np.log(1 + np.abs(x)),
        [-np.inf, np.inf]
    ]
}

INV_TRANS = {
    'linear': lambda x: x,
    'log': np.exp,
    'log10': lambda x: np.power(10, x),
    'logit': lambda x: 1 / (1 + np.exp(-x)),
    'bilog': lambda x: np.sign(x) * (np.exp(np.abs(x)) - 1)
}

def rdirichlet(dim: int, n: int = 1) -> np.ndarray:
    """Sample from the standard Dirichlet distribution

    Parameters
    ----------
    dim : int
        dimension of the sampling space
    n : int, optional
        the number of i.i.d. random sample to draw, by default 1

    Returns
    -------
    np.ndarray
        the random samples
    """
    X = -np.log(np.random.rand(n, dim))
    X /= np.sum(X, axis=1).reshape(n, -1)
    return X

def convert_inout(func):
    """Take care of the input/output of ``to_linear_scale`` and ``round`` functions
    """
    @functools.wraps(func)
    def wrapper(self, X):
        if isinstance(X, (float, int)):
            X_ = np.array([[X]])
        elif isinstance(X, (list, tuple)):
            if all([isinstance(x, (int, float)) for x in X]):
                X_ = np.array(X)
            else:
                X_ = np.array(X, dtype=object)
        elif isinstance(X, (Solution, np.ndarray)):
            X_ = X

        if len(X_.shape) == 1:
            X_ =  X_.reshape(1, -1)

        out = func(self, X_)

        # NOTE: keep the out the same type as the input
        if isinstance(X, Solution):
            X_[:, :] = out
            return X_
        elif isinstance(X, list):
            return out.tolist()
        else:
            return out
    return wrapper


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
        _n_levels : a dict of int,
            the number of unique levels for each discrete variable (as dict's keys).
        _precision : a dict of int,
            the numerical precision (granularity) of real-valued parameters,
            which is of practical relevance in real-world applications.
    """
    def __init__(
        self,
        data: List[Variable],
        random_seed: int = None
    ):
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
        self._n_levels: dict = {}
        self._precision: dict = {}
        self._scale: dict = {}
        self.r_mask: List[bool] = []
        self.i_mask: List[bool] = []
        self.d_mask: List[bool] = []
        self.id_r: Union[int, List[int]] = []
        self.id_i: Union[int, List[int]] = []
        self.id_d: Union[int, List[int]] = []

        self.random_seed : int = random_seed
        self._set_data(data)
        self.__set_type(self)

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
    def levels(self):
        return self._levels

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed):
        if seed:
            seed = int(seed)
        self._random_seed = seed
        np.random.seed(self._random_seed)

    def _check_input(
        self, bounds, var_name=None, precision=None, scale=None
    ):
        if hasattr(bounds[0], '__iter__') and not isinstance(bounds[0], str):
            bounds = [tuple(b) for b in bounds]
        else:
            bounds = [tuple(bounds)]
        dim = len(bounds)
        out: List[Dict] = [{'bounds': bounds[i]} for i in range(dim)]

        if var_name is not None:
            if isinstance(var_name, str):
                if dim > 1:
                    var_name = [var_name + str(_) for _ in range(dim)]
                else:
                    var_name = [var_name]
            assert len(var_name) == dim
            for i in range(dim):
                out[i]['name'] = var_name[i]

        if precision is not None:
            if isinstance(precision, int):
                precision = [precision] * dim if dim > 1 else [precision]
            assert len(precision) == dim
            for i in range(dim):
                out[i]['precision'] = precision[i]

        if scale is not None:
            if isinstance(scale, str):
                scale = [scale] * dim if dim > 1 else [scale]
            assert len(scale) == dim
            for i in range(dim):
                out[i]['scale'] = scale[i]
        return out

    def _check_data(self, data):
        assert all([isinstance(d, Variable) for d in data])
        names = np.asarray([var.name for var in data])
        for name, count in Counter(names).items():
            if count > 1:
                idx = np.nonzero(names == name)[0]
                _names = [name + str(i) for i in range(count)]
                for i, k in enumerate(idx):
                    data[k].name = _names[i]

    @classmethod
    def __set_type(cls, obj):
        _type = np.unique(obj.var_type)
        if len(_type) == 1:
            obj.__class__ = eval(_type[0] + 'Space')
        else:
            obj.__class__ = SearchSpace
        return obj

    def _set_data(self, data):
        self._check_data(data)
        self.data : List = data
        self.dim : int = len(self.data)
        self._bounds = [var.bounds for var in self.data]
        self._var_type = [type(v).__name__ for v in self.data]
        self._var_name = [v.name for v in self.data]
        self._set_index()
        self._set_levels()
        self._set_precision()
        self._set_scale()

    def _set_index(self):
        if len(self._var_type) != 0:
            self.r_mask = np.asarray(self._var_type) == 'Real'
            self.i_mask = np.asarray(self._var_type) == 'Integer'
            self.d_mask = np.asarray(self._var_type) == 'Discrete'
            self.id_r = np.nonzero(self.r_mask)[0]
            self.id_i = np.nonzero(self.i_mask)[0]
            self.id_d = np.nonzero(self.d_mask)[0]

    def _set_levels(self):
        """Set categorical levels for all nominal variables
        """
        if len(self.id_d) > 0:
            self._levels = {i : self._bounds[i] for i in self.id_d}
            self._n_levels = {i : len(self._bounds[i]) for i in self.id_d}
        else:
            self._levels, self._n_levels = None, None

    def _set_precision(self):
        self._precision = {
            i : self.data[i].precision for i in self.id_r \
                if self.data[i].precision is not None
        }

    def _set_scale(self):
        self._scale = {
            i : self.data[i].scale for i in self.id_r \
                if self.data[i].scale is not None
        }

    def __getitem__(self, index) -> SearchSpace:
        if isinstance(index, slice):
            data = self.data[index]
            if not isinstance(data, list):
                data = [data]
        elif isinstance(index, (list, np.ndarray)):
            data = [self.data[index[0]]] if len(index) == 1 else \
                [self.data[i] for i in index]
        else:
            data = [self.data[index]]

        return SearchSpace(data, self.random_seed)

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            self.data[index] = value
        elif isinstance(index, list):
            for i, v in zip(index, value):
                self.data[i] = v
        self._set_data(self.data)

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.var_name

    def __len__(self):
        return self.dim

    def __iter__(self):
        i = 0
        while i < self.dim:
            yield self.__getitem__(i)
            i += 1

    def __add__(self, space) -> SearchSpace:
        """Direct Sum of two `SearchSpace` instances
        """
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
        self.__set_type(self)
        return self

    def __sub__(self, space) -> SearchSpace:
        """Substraction of two `SearchSpace` instances
        """
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
        self.__set_type(self)
        return self

    def __mul__(self, N) -> SearchSpace:
        """Replicate a `SearchSpace` N times
        """
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
        msg = f'{type(self).__name__} of {self.dim} variables: \n'
        for var in self.data:
            msg += str(var) + '\n'
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
        self.__set_type(self)
        return value

    def remove(self, index: Union[int, str]) -> SearchSpace:
        if isinstance(index, str):
            _index = np.nonzero(np.array(self._var_name) == index)[0]
            if len(_index) == 0:
                raise KeyError(f"The input key {index} not found in `var_name`!")
            else:
                _index = _index[0]
        elif hasattr(index, '__iter__'):
            raise KeyError("Multiple indices are not allowed!")
        else:
            _index = index

        self.data.pop(_index)
        self._set_data(self.data)
        return self.__set_type(self)

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
        return self.__set_type(self)

    def sample(
        self,
        N: int = 1,
        method: str = 'uniform',
        h: Callable = None,
        g: Callable = None
    ) -> List:
        # TODO: to support dictionary return value
        if self.dim == 0: # in case this space is empty after slicing
            return []

        N = max(int(N), 1)
        X = np.empty((N, self.dim), dtype=object)
        X[:, self.id_r] = self.__getitem__(self.id_r).sample(N, method, h, g)
        X[:, self.id_i] = self.__getitem__(self.id_i).sample(N, method, h, g)
        X[:, self.id_d] = self.__getitem__(self.id_d).sample(N, method, h, g)
        return X.tolist()

    @convert_inout
    def round(self, X: Union[Solution, np.ndarray]):
        r_subspace = self.__getitem__(self.id_r)
        X[:, self.id_r] = r_subspace.round(X[:, self.id_r])
        return X

    @convert_inout
    def to_linear_scale(self, X: Union[Solution, np.ndarray]):
        r_subspace = self.__getitem__(self.id_r)
        X[:, self.id_r] = r_subspace.to_linear_scale(X[:, self.id_r])
        return X

    @classmethod
    def from_dict(cls, param, space_name=True):
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
        assert isinstance(param, dict)

        # construct the search space
        for i, (k, v) in enumerate(param.items()):
            bounds = v['range']
            if not hasattr(bounds[0], '__iter__') or isinstance(bounds[0], str):
                bounds = [bounds]

            N = v['N'] if 'N' in v else int(1)
            assert isinstance(N, int)
            bounds *= N
            name = k if space_name else None

            # IMPORTANT: name argument is necessary for the variable grouping
            if v['type'] in ['r', 'real']:                  # real-valued parameter
                precision = v['precision'] if 'precision' in v else None
                scale = v['scale'] if 'scale' in v else None
                space_ = RealSpace(
                    bounds, var_name=k,
                    precision=precision, scale=scale
                )
            elif v['type'] in ['i', 'int', 'integer']:      # integer-valued parameter
                space_ = IntegerSpace(bounds, var_name=k)
            elif v['type'] in ['c', 'cat', 'bool']:         # category-valued parameter
                space_ = DiscreteSpace(bounds, var_name=k)

            if i == 0:
                space = space_
            else:
                space += space_

        return space

    @classmethod
    def from_json(cls, file):
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
        with open(file, 'r') as f:
            return cls.from_dict(json.load(f))


class RealSpace(SearchSpace):
    def __init__(
        self,
        bounds: List,
        var_name: Union[str, List[str]] = 'r',
        precision: Union[int, List[int]] = None,
        scale: Union[str, List[str]] = None,
        **kwargs
    ):
        out = self._check_input(bounds, var_name, precision, scale)
        data = [Real(**_) for _ in out]
        super().__init__(data, **kwargs)

    def sample(self,
        N: int = 1,
        method: str = 'uniform',
        h: Callable = None,
        g: Callable = None
    ):
        bounds = np.array([var._bounds_transformed for var in self.data])
        lb, ub = bounds[:, 0], bounds[:, 1]

        # FIXME: this is the ad-hoc solution for simplex constraints
        if h is not None:
            X = rdirichlet(self.dim, N)
        else:
            if method == 'uniform':   # uniform random samples
                X = ((ub - lb) * rand(N, self.dim) + lb)
            elif method == 'LHS':     # Latin hypercube sampling
                if N == 1:
                    X = ((ub - lb) * rand(N, self.dim) + lb)
                else:
                    X = ((ub - lb) * lhs(self.dim, samples=N, criterion='maximin') + lb)

        X = self.round(self.to_linear_scale(X))
        return X.tolist()

    @convert_inout
    def round(self, X: Union[Solution, np.ndarray]):
        assert X.shape[1] == self.dim
        # NOTE: just in case ``dtype`` is object sometimes
        X = np.array(X, dtype=float)
        for i, var in enumerate(self.data):
            X[:, i] = var.round(X[:, i])
        return X

    @convert_inout
    def to_linear_scale(self, X: Union[Solution, np.ndarray]):
        assert X.shape[1] == self.dim
        # NOTE: just in case ``dtype`` is object sometimes
        X = np.array(X, dtype=float)
        for i, var in enumerate(self.data):
            X[:, i] = var.to_linear_scale(X[:, i])
        return X


class IntegerSpace(SearchSpace):
    def __init__(
        self,
        bounds: List,
        var_name: Union[str, List[str]] = 'i',
        **kwargs
    ):
        out = self._check_input(bounds, var_name)
        data = [Integer(**_) for _ in out]
        super().__init__(data, **kwargs)

    def sample(
        self,
        N: int = 1, method: str = 'uniform',
        h: Callable = None,
        g: Callable = None
    ):
        _bounds = np.atleast_2d(self._bounds)
        lb, ub = _bounds[:, 0], _bounds[:, 1]
        X = np.empty((N, self.dim), dtype=int)
        for i in range(self.dim):
            X[:, i] = list(map(int, randint(lb[i], ub[i], N)))
        return X.tolist()


class DiscreteSpace(SearchSpace):
    def __init__(
        self,
        bounds: List,
        var_name: Union[str, List[str]] = 'd',
        **kwargs
    ):
        out = self._check_input(bounds, var_name)
        data = [Discrete(**_) for _ in out]
        super().__init__(data, **kwargs)

    def sample(
        self,
        N: int = 1,
        method: str = 'uniform',
        h: Callable = None,
        g: Callable = None
    ):
        X = np.empty((N, self.dim), dtype=object)
        for i in range(self.dim):
            idx = randint(0, self._n_levels[i], N)
            X[:, i] = [self._levels[i][_] for _ in idx]
        return X.tolist()


class Variable(ABC):
    def __init__(
        self,
        bounds: List,
        name: str,
        default_value: Union[int, float, str] = None
    ):
        """The Variable Base Class

        Parameters
        ----------
        bounds : List
            a list/tuple contains lower and upper bounds for the real and ordinal
            and a list of levels for the Nominal
        name : str
            name of the variable
        """
        if isinstance(bounds[0], list):
            bounds = bounds[0]
        self.name: str = name
        self.bounds = tuple(bounds)
        self.default_value = default_value

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        msg = f'{self.name} -> {type(self).__name__} | range: {self.bounds}'
        if hasattr(self, 'precision') and self.precision:
            msg += f' | precision: .{self.precision}f'
        if hasattr(self, 'scale'):
            msg += f' | scale: {self.scale}'
        return msg


class Real(Variable):
    """Real-valued variable
    """
    def __init__(
        self,
        bounds,
        name: str = 'r',
        default_value: float = None,
        precision: int = None,
        scale: str = 'linear'
    ):
        assert bounds[0] < bounds[1]
        super().__init__(bounds, name, default_value)
        self.precision: int = precision
        self.scale = scale

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
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
        return (X if self.scale == 'linear' else self._inv_trans(X))

    def round(self, X):
        """Round the real-valued components of `X` to the
        corresponding numerical precision, if given
        """
        X = deepcopy(X)
        if self.precision is not None:
            X = np.round(X, self.precision)
        return X


class Discrete(Variable):
    """Discrete variable
    """
    def __init__(
        self,
        bounds,
        name: str = 'd',
        default_value: Union[int, str] = None
    ):
        bounds = self._get_unique_levels(bounds)
        super().__init__(bounds, name, default_value)

    def _get_unique_levels(self, levels):
        return sorted(list(set(levels)))


class Integer(Variable):
    """Integer variable
    """
    def __init__(
        self,
        bounds,
        name: str = 'i',
        default_value: int = None
    ):
        assert bounds[0] < bounds[1]
        super().__init__(bounds, name, default_value)


class Bool(Variable):
    """Integer variable
    """
    def __init__(
        self,
        name: str = 'b',
        default_value: int = True
    ):
        super().__init__((True, False), name, default_value)
