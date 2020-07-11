from pdb import set_trace
import numpy as np 
from BayesOpt import ContinuousSpace, OrdinalSpace, NominalSpace, from_dict, Solution

np.random.seed(1)

C = ContinuousSpace([-5, 5]) * 3  # product of the same space
I = OrdinalSpace([[-100, 100], [-5, 5]], 'heihei')
N = NominalSpace([['OK', 'A', 'B', 'C', 'D', 'E', 'A']] * 2, ['x', 'y'])

I3 = 3 * I 
print(I3.sampling())
print(I3.var_name)
print(I3.var_type)

print(C.sampling(1, 'uniform'))

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

# test for space names and save to dictionary
if 1 < 2:
    C1 = ContinuousSpace([0, 1], name='C1') 
    C2 = OrdinalSpace([-5, 5], var_name='O1') * 4
    space = C1 + C2

    d = Solution(np.random.rand(5).tolist())
    print(d)
    print(space.to_dict(d))

a = from_dict({"activation" : 
                {
                    "type" : "c",
                    "range" : ["elu", "selu", "softplus", "softsign", "relu", "tanh", 
                               "sigmoid", "hard_sigmoid", "linear"],
                    "N" : 3
                }
              })
              
print(a.var_name)
print(a.sampling(1))

a = NominalSpace(['aaa'], name='test')
print(a.sampling(3))