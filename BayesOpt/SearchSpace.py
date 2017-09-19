
class SearchSpace(object):
    def __init__(self, name):
        self.name = name

    def sampling(self, n_sample=1):
        pass

    def __iter__(self):
        pass
    def __mul__(self):
        """
        cartesian product os the search spaces
        """
        pass

class ContinuousSpace(SearchSpace):
    def __init__(self, name, bounds):
        super(ContinuousSpace, self).__init__(name)
        self.bounds = bounds
        self.dim = 1
        self.type = ['R']

class DiscreteSpace(SearchSpace):
    def __init__(self, name, levels):
        super(DiscreteSpace, self).__init__(name)
        self.levels = levels
        self.dim = 1
        self.type = ['D']

class IntegerSpace(SearchSpace):
    def __init__(self, name, bounds):
        super(IntegerSpace, self).__init__(name)
        self.bounds = bounds
        self.dim = 1
        self.type = ['I']