"""
@author: Hao Wang
@email: wangronin@gmail.com
"""
from __future__ import print_function
from pdb import set_trace

import six
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from numpy.random import randint, rand
from abc import abstractmethod
from pyDOE import lhs

# TODO: fix bugs in bounds when calling __mul__
# TODO: implement the sampling method: LHS for mixed search space
class SearchSpace(object):
    def __init__(self, bounds, var_name, name):
        """Search Space Base Class
        Arguments
        ---------
        bounds : (list of) list,
            lower and upper bound for continuous/ordinal parameter type
            categorical values for nominal parameter type.
            The dimension of the space is determined by the length of the 
            nested list
        var_name : (list of) str,
            variable name per dimension. If only a string is given for multiple 
            dimensions, variable names are created by appending counting numbers
            to the input string. 
        name : str,
            search space name. It is typically used as the grouping variable
            when converting the Solution object to dictionary, allowing for 
            vector-valued search parameters. See 'to_dict' method below.

        Attributes
        ----------
        dim : int,
            dimensinality of the search space
        bounds : list of lists
            each sub-list stores the lower and upper bound for continuous/ordinal variable
            and categorical values for nominal variable
        var_name : list of str,
            variable names per dimension 
        var_type : list of str, 
            variable type per dimension, 'C': continuous, 'N': nominal, 'O': ordinal
        C_mask : bool array,
            the mask array to index continuous variables
        id_C : int array,
            the index array for continuous variables
        """
        if hasattr(bounds[0], '__iter__') and not isinstance(bounds[0], str):
            self.bounds = [tuple(b) for b in bounds]
        else:
            self.bounds = [tuple(bounds)]
            
        self.dim = len(self.bounds)
        self.name = name
        if var_name is not None:
            if isinstance(var_name, str):
                if self.dim > 1:
                    var_name = [var_name + str(_) for _ in range(self.dim)]
                else:
                    var_name = [var_name]
            assert len(var_name) == self.dim
            self.var_name = var_name

    @abstractmethod
    def sampling(self, N=1):
        """
        The output is a list of shape (N, self.dim)
        """
        pass
    
    def _set_index(self):
        self.C_mask = np.asarray(self.var_type) == 'C'  # Continuous
        self.O_mask = np.asarray(self.var_type) == 'O'  # Ordinal
        self.N_mask = np.asarray(self.var_type) == 'N'  # Nominal 
        
        self.id_C = np.nonzero(self.C_mask)[0]
        self.id_O = np.nonzero(self.O_mask)[0]
        self.id_N = np.nonzero(self.N_mask)[0]

    def __len__(self):
        return self.dim

    def __iter__(self):
        pass

    def __add__(self, space):
        """Direct Sum of two Spaces
        """
        assert isinstance(space, SearchSpace)
        return ProductSpace(self, space)

    def __radd__(self, space):
        return self.__add__(space)

    def __mul__(self, N):
        """Replicate a Space N times
        """
        N = int(N)
        s = deepcopy(self)
        s.dim = int(self.dim * N)
        s.var_type *= N
        s.bounds *= N
        s.var_name = ['{}{}'.format(v, k) for k in range(N) for v in self.var_name]
        return s

    def __rmul__(self, N):
        return self.__mul__(N)
    
    # TODO: maybe this function should be moved to base.py
    def to_dict(self, solution):
        if self.name is None:
            return solution.to_dict() 
        else:
            return {self.name : solution.tolist()}


# TODO: maybe implement the scalar multiplication for ProductSpace
class ProductSpace(SearchSpace):
    """Cartesian product of the search spaces
    """
    def __init__(self, space1, space2):
        # TODO: avoid recursion here
        self.dim = space1.dim + space2.dim
        # check coincides of variable names
        self.var_name = space1.var_name + space2.var_name             
        self.bounds = space1.bounds + space2.bounds
        self.var_type = space1.var_type + space2.var_type
        self._sub_space1 = deepcopy(space1)
        self._sub_space2 = deepcopy(space2)
        self._set_index()

        # setup the space names
        nameL = space1.name if isinstance(space1, ProductSpace) else [space1.name] 
        nameR = space2.name if isinstance(space2, ProductSpace) else [space2.name]
        self.name = nameL + nameR

        if len(self.id_N) > 0:  # set levels only if Nominal variables present
            self.levels = OrderedDict([(i, self.bounds[i]) for i in self.id_N]) 
    
    def sampling(self, N=1, method='uniform'):
        # TODO: should recursion be avoided here?
        a = self._sub_space1.sampling(N, method)
        b = self._sub_space2.sampling(N, method)
        return [a[i] + b[i] for i in range(N)]
    
    def to_dict(self, solution):
        """
        Save a Solution instance to dict, grouped by sub-spaces
        This is meant for vector-valued parameters for the configuration 
        """
        id1 = list(range(self._sub_space1.dim))
        id2 = list(range(self._sub_space1.dim, self.dim))
        L = solution[id1] if len(solution.shape) == 1 else solution[:, id1]
        R = solution[id2] if len(solution.shape) == 1 else solution[:, id2]
        return {**self._sub_space1.to_dict(L), **self._sub_space2.to_dict(R)}

    def __mul__(self, space):
        raise ValueError('Unsupported operation')

    def __rmul__(self, space):
        raise ValueError('Unsupported operation')


class ContinuousSpace(SearchSpace):
    """Continuous Search Space Class
    """
    def __init__(self, bounds, var_name=None, name=None):
        super(ContinuousSpace, self).__init__(bounds, var_name, name)
        if not hasattr(self, 'var_name'):
            self.var_name = ['r' + str(i) for i in range(self.dim)]

        self.var_type = ['C'] * self.dim
        self._bounds = np.atleast_2d(self.bounds).T
        assert all(self._bounds[0, :] < self._bounds[1, :])
        self._set_index()

    def __mul__(self, N):
        s = super(ContinuousSpace, self).__mul__(N)
        s._bounds = np.tile(s._bounds, (1, N))
        s._set_index()
        return s
    
    def sampling(self, N=1, method='uniform'):
        lb, ub = self._bounds
        if method == 'uniform':   # uniform random samples
            return ((ub - lb) * rand(N, self.dim) + lb).tolist()
        elif method == 'LHS':     # Latin hypercube sampling
            if N == 1:
                raise ValueError('LHS: number of sample cannot be 1')
            else:
                return ((ub - lb) * lhs(self.dim, samples=N, criterion='cm') + lb).tolist()


class NominalSpace(SearchSpace):
    """Nominal (discrete) Search Space Class
    """
    def __init__(self, levels, var_name=None, name=None):
        super(NominalSpace, self).__init__(levels, var_name, name)
        if not hasattr(self, 'var_name'):
            self.var_name = ['d' + str(i) for i in range(self.dim)]

        self.var_type = ['N'] * self.dim
        self._levels = [np.array(b) for b in self.bounds]
        self._n_levels = [len(l) for l in self._levels]
        self._set_index()
        
    def __mul__(self, N):
        s = super(NominalSpace, self).__mul__(N)
        s.levels = OrderedDict([(i, s.bounds[i]) for i in range(s.dim)])
        s._levels = s._levels * N
        s._n_levels = s._n_levels * N
        s._set_index()
        return s
    
    def sampling(self, N=1, method='uniform'):
        # TODO: adding LHS sampling here
        res = np.empty((N, self.dim), dtype=object)
        for i in range(self.dim):
            res[:, i] = self._levels[i][randint(0, self._n_levels[i], N)]
        return res.tolist()


# TODO: add integer multiplication for OrdinalSpace
class OrdinalSpace(SearchSpace):
    """Ordinal (Integer) Search Space
    """
    def __init__(self, bounds, var_name=None, name=None):
        super(OrdinalSpace, self).__init__(bounds, var_name, name)
        if not hasattr(self, 'var_name'):
            self.var_name = ['i' + str(i) for i in range(self.dim)]

        self.var_type = ['O'] * self.dim
        self._lb, self._ub = zip(*self.bounds)        # for sampling
        assert all(np.array(self._lb) < np.array(self._ub))
        self._set_index()

    def __mul__(self, N):
        s = super(OrdinalSpace, self).__mul__(N)
        s._lb, s._ub = s._lb * N, s._ub * N
        s._set_index()
        return s
    
    def sampling(self, N=1, method='uniform'):
        # TODO: adding LHS sampling here
        res = np.zeros((N, self.dim), dtype=int)
        for i in range(self.dim):
            res[:, i] = list(map(int, randint(self._lb[i], self._ub[i], N)))
        return res.tolist()


def from_dict(param, space_name=True):
    """Create a seach space from input dictionary
    """
    assert isinstance(param, dict)
    # construct the search space
    for i, (k, v) in enumerate(param.items()):
        bounds = v['range']
        if not hasattr(bounds[0], '__iter__') or isinstance(bounds[0], str):
            bounds = [bounds]

        bounds *= v['N']
        name = k if space_name else None

        # IMPORTANT: name argument is necessary for the variable grouping
        if v['type'] == 'r':        # real-valued parameter
            space_ = ContinuousSpace(bounds, var_name=k, name=name) 
        elif v['type'] == 'i':      # integer-valued parameter
            space_ = OrdinalSpace(bounds, var_name=k, name=name)
        elif v['type'] == 'c':      # category-valued parameter
            space_ = NominalSpace(bounds, var_name=k, name=name) 
        
        if i == 0:
            space = space_
        else:
            space += space_
    
    return space

def from_json(param):
    pass

if __name__ == '__main__':
    np.random.seed(1)

    C = ContinuousSpace([-5, 5]) * 3  # product of the same space
    I = OrdinalSpace([[-100, 100], [-5, 5]], 'heihei')
    N = NominalSpace([['OK', 'A', 'B', 'C', 'D', 'E']] * 2, ['x', 'y'])

    I3 = 3 * I 
    print(I3.sampling())
    print(I3.var_name)
    print(I3.var_type)

    print(C.sampling(1, 'uniform'))

    # cartesian product of heterogeneous spaces
    space = C + I + N 
    print(space.sampling(10))
    print(space.bounds)
    print(space.var_name)
    print(space.var_type)

    print((C * 2).var_name)
    print((N * 3).sampling(2))

    C = ContinuousSpace([[0, 1]] * 2, var_name='weight')
    print(C.var_name)

    # test for space names and save to dictionary
    if 1 < 2:
        from .base import Solution

        C1 = ContinuousSpace([0, 1], name='C1') 
        C2 = OrdinalSpace([-5, 5], var_name='O1') * 4
        space = C1 + C2

        d = Solution(np.random.rand(5).tolist())
        print(d)
        print(space.to_dict(d))

    a = from_dict({"activation" : 
                    {
                        "type" : "c",
                        "range" : ["elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"],
                        "N" : 3
                    }
                  })
    print(a.var_name)
    print(a.sampling(1))
