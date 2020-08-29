import pytest, sys

from pdb import set_trace
from copy import deepcopy
import numpy as np 

sys.path.insert(0, '../')
from BayesOpt import ContinuousSpace, OrdinalSpace, NominalSpace, from_dict, Solution

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

def test_sampling():
    C = ContinuousSpace([-5, 5], precision=1) * 3 
    I = OrdinalSpace([[-100, 100], [-5, 5]], 'heihei')
    N = NominalSpace([['OK', 'A', 'B', 'C', 'D', 'E', 'A']] * 2, ['x', 'y'])

    C2 = ContinuousSpace([[-5, 5]] * 3, precision=[2, None, 3])
    C2.sampling(3)

    S = N + I + C2
    S2 = N + I + ContinuousSpace([[-5, 5]] * 3)
    X = S2.sampling(5)

    print(X)
    X2 = S.round(X)
    print(X2)

    I3 = 3 * I 
    print(I3.sampling())
    print(I3.var_name)
    print(I3.var_type)
    print(C.sampling(1, 'uniform'))

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

def test_to_dict():
    # test for space names and save to dictionary
    C1 = ContinuousSpace([0, 1], name='C1') 
    C2 = OrdinalSpace([-5, 5], var_name='O1') * 4
    space = C1 + C2

    d = Solution(np.random.rand(5).tolist())
    print(d)
    print(space.to_dict(d))

    C3 = ContinuousSpace([0, 1]) * 10
    x = Solution(C3.sampling(20))
    C3.to_dict(x)

def test_from_dict():
    a = from_dict(
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