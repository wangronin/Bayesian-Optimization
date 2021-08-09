from __future__ import annotations

import re
import sys
import warnings
from abc import ABC
from copy import copy, deepcopy
from itertools import chain, combinations
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.random import randint
from scipy.special import logit

__authors__ = ["Hao Wang"]

MAX = sys.float_info.max
TRANS = {
    "linear": [lambda x: x, [-MAX, MAX]],
    "log": [np.log, [1e-300, MAX]],
    "log10": [np.log10, [1e-300, MAX]],
    "logit": [logit, [1e-300, 1]],
    "bilog": [lambda x: np.sign(x) * np.log(1 + np.abs(x)), [-MAX, MAX]],
}
INV_TRANS = {
    "linear": lambda x: x,
    "log": np.exp,
    "log10": lambda x: np.power(10, x),
    "logit": lambda x: 1 / (1 + np.exp(-x)),
    "bilog": lambda x: np.sign(x) * (np.exp(np.abs(x)) - 1),
}


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
        self.set_default_value(default_value)
        self.add_conditions(conditions, action)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        msg = f"{self.name} -> {type(self).__name__} | range: {self.bounds}"
        if self.default_value is not None:
            msg += f" | default: {self.default_value}"
        return msg

    def __contains__(self, x: Union[float, str, object]) -> bool:
        pass

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

    def copyfrom(self, var: Variable):
        self.__dict__.update(**deepcopy(var.__dict__))

    def set_default_value(self, value):
        """validate the default value first"""
        if value is not None:
            assert self.__contains__(value)
        self.default_value = value

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

    def __contains__(self, x: Union[float, str]) -> bool:
        return self.bounds[0] <= x <= self.bounds[1]

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
        bounds = list(self.bounds)

        if (bounds[0] < _range[0]) or (bounds[0] > _range[1]):
            bounds[0] = _range[0]
            warnings.warn(
                f"lower bound {bounds[0]} not in the working "
                f"range of the given scale {self._scale} is set to the default value {_range[0]}"
            )
            # raise ValueError(
            #     f"lower bound {self.bounds[0]} not in the working "
            #     f"range of the given scale {self._scale}"
            # )

        if (bounds[1] < _range[0]) or (bounds[1] > _range[1]):
            bounds[1] = _range[1]
            warnings.warn(
                f"upper bound {bounds[1]} not in the working "
                f"range of the given scale {self._scale} is set to the default value {_range[1]}"
            )
            # raise ValueError(
            #     f"upper bound {self.bounds[1]} is invalid for the given scale {self._scale}"
            # )
        self.bounds = tuple(bounds)
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
        # get rid of duplicated levels
        bounds = list(dict.fromkeys(bounds))
        # map discrete values (bounds) to integers for sampling
        self._map_func: callable = None
        self._size: int = None
        super().__init__(bounds, *args, **kwargs)

    def __contains__(self, x: Union[int, str]) -> bool:
        return x in self.bounds

    def __hash__(self):
        return hash((self.name, self.bounds, self.default_value))

    def sample(self, N: int = 1, **kwargs) -> List:
        return list(map(self._map_func, randint(0, self._size, N)))


class Discrete(_Discrete):
    """Discrete variable, whose values should come with a linear order"""

    def __init__(self, bounds, name: str = "d", default_value: Union[int, str] = None, **kwargs):
        super().__init__(bounds, name, default_value, **kwargs)
        self._map_func = lambda i: self.bounds[i]
        self._size = len(self.bounds)


class Subset(Discrete):
    """A discrete variable created by enumerating all subsets of the input `bounds`"""

    def __init__(self, bounds, name: str = "s", default_value: Union[int, str] = None, **kwargs):
        self._bounds = bounds
        bounds = list(
            chain.from_iterable(map(lambda r: combinations(bounds, r), range(1, len(bounds) + 1)))
        )
        super().__init__(bounds, name, default_value, **kwargs)

    def __str__(self):
        msg = f"{self.name} -> {type(self).__name__} | range: 2 ^ {self._bounds}"
        if self.default_value is not None:
            msg += f" | default: {self.default_value}"
        return msg


class Ordinal(_Discrete):
    """A generic ordinal variable, whose values should come with a linear order"""

    def __init__(self, bounds, name: str = "ordinal", default_value: int = None, **kwargs):
        super().__init__(bounds, name, default_value, **kwargs)
        self._map_func = lambda i: self.bounds[i]
        self._size = len(self.bounds)


class Integer(_Discrete):
    """Integer variable"""

    def __init__(
        self,
        bounds: Tuple[int],
        name: str = "i",
        default_value: int = None,
        step: Optional[Union[int, float]] = 1,
    ):
        super().__init__(bounds, name, default_value)
        assert len(self.bounds) == 2
        assert self.bounds[0] < self.bounds[1]
        assert all(map(lambda x: isinstance(x, (int, float)), self.bounds))
        self.step = step
        self._map_func = lambda i: self.bounds[0] + i * self.step
        self._size = int(np.floor((self.bounds[1] - self.bounds[0]) / self.step) + 1)

    def __contains__(self, x: int) -> bool:
        return self.bounds[0] <= x <= self.bounds[1]

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
