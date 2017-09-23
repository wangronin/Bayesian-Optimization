import six
from abc import abstractmethod
import pdb
import numpy as np
from numpy.random import randint, rand

class SearchSpace(object):
    def __init__(self, var_name, bounds):
        self.var_name = [var_name] if isinstance(var_name, six.string_types) else var_name
        self.dim = len(self.var_name)

        if not hasattr(bounds[0], '__iter__'):
            self.bounds = [tuple(bounds)]
        else:
            self.bounds = [tuple(b) for b in bounds]
        assert len(self.bounds) == self.dim

    @abstractmethod
    def sampling(self, N=1):
        pass

    def get_continous(self):
        if not hasattr(self, 'C_mask'):
            self.C_mask = self.var_type == 'C'
        return np.array(self.var_name)[self.C_mask].tolist()
    
    def get_norminal(self):
        if not hasattr(self, 'N_mask'):
            self.N_mask = self.var_type == 'N'
        return np.array(self.var_name)[self.N_mask].tolist()
    
    def get_ordinal(self):
        if not hasattr(self, 'O_mask'):
            self.O_mask = self.var_type == 'O'
        return np.array(self.var_name)[self.O_mask].tolist()

    def get_levels(self):
        if hasattr(self, '_levels'):
            return self._levels
        else:
            return []

    def __len__(self):
        return self.dim

    def __iter__(self):
        pass

    def __mul__(self, space):
        return ProductSpace(self, space)

class ProductSpace(SearchSpace):
    """Cartesian product of the search spaces
    """
    def __init__(self, space1, space2):
        # TODO: avoid recursion here
        self.dim = space1.dim + space2.dim
        self.var_name = space1.var_name + space2.var_name
        self.bounds = space1.bounds + space2.bounds
        self.var_type = np.r_[space1.var_type, space2.var_type]
        self._sub_space1 = space1
        self._sub_space2 = space2

        self.C_mask = self.var_type == 'C'
        self.N_mask = self.var_type == 'N'
        self.O_mask = self.var_type == 'O'
        
        id_N = np.nonzero(self.N_mask)[0]
        self._levels = [self.bounds[i] for i in id_N]
    
    def sampling(self, N=1):
        a = self._sub_space1.sampling(N)
        b = self._sub_space2.sampling(N)
        return [a[i] + b[i] for i in range(N)]

class ContinuousSpace(SearchSpace):
    """Continuous search spaces
    """
    def __init__(self, var_name, bounds):
        super(ContinuousSpace, self).__init__(var_name, bounds)
        self.var_type = np.array(['C'] * self.dim)
        self._bounds = np.atleast_2d(self.bounds).T
        assert all(self._bounds[0, :] < self._bounds[1, :])

    def sampling(self, N=1, method='uniform'):
        lb, ub = self._bounds
        return ((ub - lb) * rand(N, self.dim) + lb).tolist()

class NominalSpace(SearchSpace):
    """Nominal search spaces
    """
    def __init__(self, var_name, levels):
        super(NominalSpace, self).__init__(var_name, levels)
        self.var_type = np.array(['N'] * self.dim)
        self._levels = np.array(levels)
        self._n_levels = len(levels)
    
    def sampling(self, N=1):
        res = np.empty((N, self.dim), dtype=object)
        for i in range(self.dim):
            res[:, i] = self._levels[randint(0, self._n_levels, N)]
        return res.tolist()
            
class OrdinalSpace(SearchSpace):
    """Ordinal (Integer) the search spaces
    """
    def __init__(self, var_name, bounds):
        super(OrdinalSpace, self).__init__(var_name, bounds)
        self.var_type = np.array(['O'] * self.dim)
        self._lb, self._ub = zip(*self.bounds)
        assert all(np.array(self._lb) < np.array(self._ub))
    
    def sampling(self, N=1):
        res = np.zeros((N, self.dim), dtype=int)
        for i in range(self.dim):
            res[:, i] = randint(self._lb[i], self._ub[i], N)
        return res.tolist()

if __name__ == '__main__':

    C = ContinuousSpace(['x1', 'x2'], [[-5, 5], [-5, 5]])
    I = OrdinalSpace(['x3'], [-100, 100])
    N = NominalSpace(['x4'], ['OK', 'A', 'B', 'C', 'D', 'E'])

    print C.get_continous()
    space = C * I * N
    print space.sampling(10)

    print space.get_continous()
    print space.get_norminal()
    print space.get_ordinal()