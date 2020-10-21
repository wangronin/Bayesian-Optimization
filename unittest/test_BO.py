import numpy as np
import sys, os
import pytest
sys.path.insert(0, '../')

from bayes_optim import ParallelBO, BO, ContinuousSpace, OrdinalSpace, NominalSpace
from bayes_optim.Surrogate import trend, GaussianProcess, RandomForest

np.random.seed(123)

def test_pickling():
    dim = 5
    lb, ub = -1, 5

    def fitness(x):
        x = np.asarray(x)
        return np.sum(x ** 2)

    space = ContinuousSpace([lb, ub]) * dim

    mean = trend.constant_trend(dim, beta=None)
    thetaL = 1e-10 * (ub - lb) * np.ones(dim)
    thetaU = 10 * (ub - lb) * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

    model = GaussianProcess(
        mean=mean, corr='squared_exponential',
        theta0=theta0, thetaL=thetaL, thetaU=thetaU,
        nugget=0, noise_estim=False,
        optimizer='BFGS', wait_iter=3, random_start=dim,
        likelihood='concentrated', eval_budget=100 * dim
    )
    opt = BO(
        search_space=space,
        obj_fun=fitness,
        model=model,
        DoE_size=5,
        max_FEs=10,
        verbose=True,
        n_point=1
    )
    opt.step()

    opt.save('test')
    opt = BO.load('test')

    print(opt.run())
    os.remove('test')

def test_pickling2():
    dim = 5
    lb, ub = -1, 5

    def fitness(x):
        x = np.asarray(x)
        return np.sum(x ** 2)

    space = ContinuousSpace([lb, ub]) * dim

    mean = trend.constant_trend(dim, beta=None)
    thetaL = 1e-10 * (ub - lb) * np.ones(dim)
    thetaU = 10 * (ub - lb) * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

    model = GaussianProcess(
        mean=mean, corr='squared_exponential',
        theta0=theta0, thetaL=thetaL, thetaU=thetaU,
        nugget=0, noise_estim=False,
        optimizer='BFGS', wait_iter=3, random_start=dim,
        likelihood='concentrated', eval_budget=100 * dim
    )
    opt = BO(
        search_space=space,
        obj_fun=fitness,
        model=model,
        DoE_size=5,
        max_FEs=10,
        verbose=True,
        n_point=1,
        logger='log'
    )
    opt.save('test')
    opt = BO.load('test')

    print(opt.run())

    os.remove('test')
    os.remove('log')

def test_continuous():
    dim = 5
    lb, ub = -1, 5

    def fitness(x):
        x = np.asarray(x)
        return np.sum(x ** 2)

    space = ContinuousSpace([lb, ub]) * dim

    mean = trend.constant_trend(dim, beta=None)
    thetaL = 1e-10 * (ub - lb) * np.ones(dim)
    thetaU = 10 * (ub - lb) * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

    model = GaussianProcess(
        mean=mean, corr='squared_exponential',
        theta0=theta0, thetaL=thetaL, thetaU=thetaU,
        nugget=0, noise_estim=False,
        optimizer='BFGS', wait_iter=3, random_start=dim,
        likelihood='concentrated', eval_budget=100 * dim
    )

    opt = BO(
        search_space=space,
        obj_fun=fitness,
        model=model,
        DoE_size=5,
        max_FEs=10,
        verbose=True,
        n_point=1
    )
    print(opt.run())

@pytest.mark.parametrize("eval_type", ['list', 'dict', 'dataframe'])  # type: ignore
def test_mix_space(eval_type):
    dim_r = 2  # dimension of the real values
    def obj_fun(x):
        x_r = np.array([x['continuous_%d'%i] for i in range(dim_r)])
        x_i = x['ordinal']
        x_d = x['nominal']
        _ = 0 if x_d == 'OK' else 1
        return np.sum(x_r ** 2) + abs(x_i - 10) / 123. + _ * 2

    search_space = ContinuousSpace([-5, 5], var_name='continuous') * dim_r + \
        OrdinalSpace([5, 15], var_name='ordinal') + \
        NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], var_name='nominal')

    model = RandomForest(levels=search_space.levels)

    opt = ParallelBO(
        search_space=search_space,
        obj_fun=obj_fun,
        model=model,
        max_FEs=6,
        DoE_size=3,    # the initial DoE size
        eval_type=eval_type,
        acquisition_fun='MGFI',
        acquisition_par={'t' : 2},
        n_job=3,       # number of processes
        n_point=3,     # number of the candidate solution proposed in each iteration
        verbose=True   # turn this off, if you prefer no output
    )
    xopt, fopt, stop_dict = opt.run()

    print('xopt: {}'.format(xopt))
    print('fopt: {}'.format(fopt))
    print('stop criteria: {}'.format(stop_dict))