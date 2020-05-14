import numpy as np
from BayesOpt.base import Solution

# test for 2D solution
A, B = np.random.randn(5, 3).tolist(), ['simida', 'niubia', 'bang', 'GG', 'blyat']
s = Solution([A[i] + [B[i]] for i in range(5)], 
             verbose=True, fitness=[0] * 2, fitness_name=['f', 'penalty'], n_obj=2)

print(s.to_dict(orient='index'))
print(s.to_dict(orient='var'))

print(s)
print(s[0:1])
print(s[0, 0:3])
print(s[:, 0])
print(s[0:2][0, 0:2])
print(s[0][0:2])

# test for pickling
if 11 < 2:
    import dill
    a = dill.dumps(s)
    s2 = dill.loads(a)
    print(s2)

s[:, 0] = np.asarray(['wa'] * 5).reshape(-1, 1)
print(s)

a = s[0]
a.fitness = 3
print(s)

s[0].fitness = 2
print(s)

s[1].fitness = [1, 2]
s[2:4].fitness = [[3, 4], [5, 6]]
s[0:2].n_eval += 1
print(s)

# s += s[3:5]
print(s + s[3:5])
print(s[0:2] + s[3:5])

# # test saving to csv
if 11 < 2:
    s.to_csv('test.csv', header=True, show_attr=True, index=True)

# test for 1D solution
ss = Solution(np.random.randn(3, 5))

# print(ss * 2)
# print(ss[0])
# print(ss[0:5])
a = ss[0]
a.fitness = 3  # TODO: change the fitness to a 1-d array
print(ss)

# test for pickling
# np.save('test', s)

# test for one-dimensional case
if 11 < 2:
    d = Solution([np.random.randint(0, 100, size=(1,)).tolist() for i in range(10)], 
                  index=list(range(10)))
    print(d)
    for dd in d:
        print(dd.tolist())