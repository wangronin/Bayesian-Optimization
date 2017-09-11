
class SearchSpace(object):
    def __init__(self):
        pass

    def sampling(self, n_sample=1):
        pass

    def __mul__(self):
        """
        cartesian product os the search spaces
        """
        pass

class ContinuousSpace(SearchSpace):
    pass

class DiscreteSpace(SearchSpace):
    pass

class IntegerSpace(SearchSpace):
    pass