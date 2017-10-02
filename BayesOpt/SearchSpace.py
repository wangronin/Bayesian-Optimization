import pdb

import six
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from numpy.random import randint, rand
from abc import abstractmethod
from pyDOE import lhs

class SearchSpace(object):
    def __init__(self, bounds, var_name):
        if not hasattr(bounds[0], '__iter__'):
            self.bounds = [tuple(bounds)]
        else:
            self.bounds = [tuple(b) for b in bounds]
        self.dim = len(self.bounds)
        if var_name is not None:
            var_name = [var_name] if isinstance(var_name, six.string_types) else var_name
            self.var_name = np.array(var_name)

    @abstractmethod
    def sampling(self, N=1):
        """
        The output is a list of shape (N, self.dim)
        """
        pass

    def __len__(self):
        return self.dim

    def __iter__(self):
        pass

    def __mul__(self, space):
        if isinstance(space, SearchSpace):
            return ProductSpace(self, space)
        else:  # all the other types of input should be handled in the derived classes
            raise ValueError('multiply SearchSpace to an invalid type')

    def __rmul__(self, space):
        return self.__mul__(space)

class ProductSpace(SearchSpace):
    """Cartesian product of the search spaces
    """
    def __init__(self, space1, space2):
        # TODO: avoid recursion here
        self.dim = space1.dim + space2.dim
        # check coincides of variable names
        self.var_name = np.r_[space1.var_name, space2.var_name]
        self.bounds = space1.bounds + space2.bounds
        self.var_type = np.r_[space1.var_type, space2.var_type]
        self._sub_space1 = deepcopy(space1)
        self._sub_space2 = deepcopy(space2)

        self.C_mask = self.var_type == 'C'  # Continuous
        self.O_mask = self.var_type == 'O'  # Ordinal
        self.N_mask = self.var_type == 'N'  # Nominal 

        self.id_C = np.nonzero(self.C_mask)[0]
        self.id_O = np.nonzero(self.O_mask)[0]
        self.id_N = np.nonzero(self.N_mask)[0]
        
        self.levels = OrderedDict([(i, self.bounds[i]) for i in self.id_N])
    
    def sampling(self, N=1):
        # TODO: should recursion be avoided here?
        a = self._sub_space1.sampling(N)
        b = self._sub_space2.sampling(N)
        return [a[i] + b[i] for i in range(N)]

    def __rmul__(self, space):
        raise ValueError('Not suppored operation')

class ContinuousSpace(SearchSpace):
    """Continuous search space
    """
    def __init__(self, bounds, var_name=None):
        super(ContinuousSpace, self).__init__(bounds, var_name)
        if not hasattr(self, 'var_name'):
            self.var_name = np.array(['r' + str(i) for i in range(self.dim)])
        self.var_type = np.array(['C'] * self.dim)
        self._bounds = np.atleast_2d(self.bounds).T
        assert all(self._bounds[0, :] < self._bounds[1, :])
    
    def __mul__(self, N):
        if isinstance(N, SearchSpace):
            return super(ContinuousSpace, self).__mul__(N)
        else: # multiple times the same space
            self.dim = int(self.dim * N)
            self.var_type = np.repeat(self.var_type, N)
            self.var_name = np.array(['r' + str(i) for i in range(self.dim)])
            self.bounds = self.bounds * N
            self._bounds = np.tile(self._bounds, (1, N))
            return self
    
    def __rmul__(self, N):
        return self.__mul__(N)

    def sampling(self, N=1, method='uniform'):
        lb, ub = self._bounds
        if method == 'uniform':   # uniform random samples
            return ((ub - lb) * rand(N, self.dim) + lb).tolist()
        elif method == 'LHS':     # Latin hypercube sampling
            return ((ub - lb) * lhs(self.dim, samples=N, criterion='cm') + lb).tolist()

class NominalSpace(SearchSpace):
    """Nominal search spaces
    """
    def __init__(self, levels, var_name=None):
        super(NominalSpace, self).__init__(levels, var_name)
        if not hasattr(self, 'var_name'):
            self.var_name = np.array(['d' + str(i) for i in range(self.dim)])
        self.var_type = np.array(['N'] * self.dim)
        self._levels = np.unique(np.array(levels))
        self._n_levels = len(self._levels)
        # public attribute: dict of levels
        self.levels = OrderedDict([(i, self.bounds[i]) for i in range(self.dim)])
    
    def sampling(self, N=1):
        res = np.empty((N, self.dim), dtype=object)
        for i in range(self.dim):
            res[:, i] = self._levels[randint(0, self._n_levels, N)]
        return res.tolist()
            
class OrdinalSpace(SearchSpace):
    """Ordinal (Integer) the search spaces
    """
    def __init__(self, bounds, var_name=None):
        super(OrdinalSpace, self).__init__(bounds, var_name)
        if not hasattr(self, 'var_name'):
            self.var_name = np.array(['i' + str(i) for i in range(self.dim)])
        self.var_type = np.array(['O'] * self.dim)
        self._lb, self._ub = zip(*self.bounds)
        assert all(np.array(self._lb) < np.array(self._ub))
    
    def sampling(self, N=1):
        res = np.zeros((N, self.dim), dtype=int)
        for i in range(self.dim):
            res[:, i] = randint(self._lb[i], self._ub[i], N)
        return res.tolist()

if __name__ == '__main__':
    np.random.seed(1)

    C = ContinuousSpace([[-5, 5]]) * 2  # product of the same space
    I = OrdinalSpace([-100, 100], 'x3')
    N = NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E'])

    print C.sampling(3, 'LHS')

    # cartesian product of heterogeneous spaces
    space = C * I * N 
    print space.sampling(10)