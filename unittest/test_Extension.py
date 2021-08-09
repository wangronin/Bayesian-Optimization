import sys

import numpy as np

sys.path.insert(0, "../")
from bayes_optim import fmin

np.random.seed(42)


def test_fmin():
    def f(x):
        x = np.asarray(x)
        return np.sum(x ** 2)

    minimum = fmin(f, [-5] * 2, [5] * 2, seed=42, max_FEs=30, verbose=False)
    assert len(minimum) == 5
    assert len(minimum[0]) == 2
    # assert np.isclose(minimum[1], 0.007165794451494286)
    # assert all(np.isclose(minimum[0], [-0.04300030341296269, 0.07291617350003657]))

    # test warm starting
    X = np.random.rand(10, 2) * 10 - 5
    y = [f(x) for x in X]
    minimum = fmin(f, [-5] * 2, [5] * 2, x0=X, y0=y, max_FEs=20, verbose=False)
    assert minimum[2] == 20

    minimum = fmin(f, [-5] * 2, [5] * 2, x0=X, max_FEs=5, verbose=False)
    assert minimum[2] == 5


# def test_multi_acquisition():
# dim_r = 2  # dimension of the real values
# def obj_fun(x):
#     x_r = np.array([x['continuous_%d'%i] for i in range(dim_r)])
#     x_i = x['ordinal']
#     x_d = x['nominal']
#     _ = 0 if x_d == 'OK' else 1
#     return np.sum(x_r ** 2) + abs(x_i - 10) / 123. + _ * 2

# search_space = RealSpace([-5, 5], var_name='continuous') * dim_r + \
#     IntegerSpace([5, 15], var_name='ordinal') + \
#     DiscreteSpace(['OK', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], var_name='nominal')

# model = RandomForest(levels=search_space.levels)

# opt = MultiAcquisitionBO(
#     search_space=search_space,
#     obj_fun=obj_fun,
#     model=model,
#     max_FEs=8,
#     DoE_size=4,    # the initial DoE size
#     eval_type='dict',
#     n_job=4,       # number of processes
#     n_point=4,     # number of the candidate solution proposed in each iteration
#     verbose=True   # turn this off, if you prefer no output
# )

# xopt, fopt, stop_dict = opt.run()
# print('xopt: {}'.format(xopt))
# print('fopt: {}'.format(fopt))
# print('stop criteria: {}'.format(stop_dict))
