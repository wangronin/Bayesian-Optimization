import sys
import re
import numpy as np

sys.path.insert(0, '../')
from bayes_optim import ContinuousSpace, OrdinalSpace, NominalSpace, Solution, SearchSpace

np.random.seed(1)

def test_NominalSpace():
    S = NominalSpace(
        [['OK', 'A', 'B', 'C', 'D', 'E', 'A']] * 2, ['x', 'y']
    )
    assert all(
        set(v) == set(['OK', 'A', 'B', 'C', 'D', 'E']) for k, v in S.levels.items()
    )

    S = NominalSpace([['A'] * 3, 'B', ['x', 'y']])
    assert set(S.levels[2]) == set(['x', 'y'])

    S = NominalSpace(['x', 'y', 'z'])
    assert set(S.levels[0]) == set(['x', 'y', 'z'])

def test_precision():
    C = ContinuousSpace([-5, 5], precision=2) * 3
    X = C.round(C.sampling(1, method='LHS'))
    X = [re.sub(r'^-?\d+\.(\d+)$', r'\1', str(_)) for _ in X[0]]
    assert all([len(x) <= 2 for x in X])

    X = C.round(C.sampling(1, method='uniform'))
    X = [re.sub(r'^-?\d+\.(\d+)$', r'\1', str(_)) for _ in X[0]]
    assert all([len(x) <= 2 for x in X])

    X = np.random.rand(2, 3) * 10 - 5
    assert isinstance(C.round(X), np.ndarray)

    X_ = C.round(X.tolist())
    assert isinstance(X_, list)
    assert isinstance(X_[0], list)
    assert np.all(np.array(X_) == C.round(X))

def test_scale():
    C = ContinuousSpace([1, 5], scale='log')
    assert C.bounds[0][0] == 0

    C = ContinuousSpace([0.5, 0.8], scale='logit')
    assert C.bounds[0][0] == 0

    C = ContinuousSpace([-1, 1], scale='bilog')
    assert C.bounds[0][0] == -np.log(2)
    assert C.bounds[0][1] == np.log(2)

    C = ContinuousSpace([-1, 1], scale='bilog') * 2
    X = np.array([-np.log(2), np.log(2)])
    a = C.to_linear_scale(X)
    assert all(a == np.array([-1, 1]))

def test_sampling():
    C = ContinuousSpace([-5, 5]) * 3
    I = OrdinalSpace([[-100, 100], [-5, 5]], 'heihei')
    N = NominalSpace([['OK', 'A', 'B', 'C', 'D', 'E', 'A']] * 2, ['x', 'y'])

    S = N + I + C
    S.sampling(5, method='LHS')
    S.sampling(5, method='uniform')

def test_ProductSpace():
    C = ContinuousSpace([-5, 5], precision=1) * 3  # product of the same space
    I = OrdinalSpace([[-100, 100], [-5, 5]], 'heihei')
    N = NominalSpace([['OK', 'A', 'B', 'C', 'D', 'E', 'A']] * 2, ['x', 'y'])

    space = C + C + C
    print(space.sampling(2))

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

def test_from_dict():
    a = SearchSpace.from_dict(
        {
            "activation" :
            {
                "type" : "c",
                "range" : [
                    "elu", "selu", "softplus", "softsign", "relu", "tanh",
                    "sigmoid", "hard_sigmoid", "linear"
                ],
                "N" : 3
            }
        }
    )

    print(a.var_name)
    print(a.sampling(1))

    a = NominalSpace(['aaa'], name='test')
    print(a.sampling(3))