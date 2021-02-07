import sys
import pytest
import numpy as np
sys.path.insert(0, '../')

from bayes_optim import BO, RealSpace, IntegerSpace, DiscreteSpace
from bayes_optim.surrogate import RandomForest
from bayes_optim.acquisition_optim import OnePlusOne_Cholesky_CMA

def obj_fun2(x):
    return (x['pc'] - 0.2) ** 2 + x['mu'] + x['lambda'] + np.abs(x['p'] - 0.7)

def obj_fun(x):
    return np.sum(np.array(x) ** 2) +  5 * np.sum(np.array(x)) + 10

def h(x):
    return np.sum(x) - 1

def g(x):
    return [-x['pc'], x['mu'] - 1.9]

@pytest.mark.skip(reason="OnePlusOne_Cholesky_CMA does not work this constraints yet..")
def test_BO_equality():
    search_space = RealSpace([0, 1]) * 2
    model = RandomForest(levels=search_space.levels)
    xopt, _, __ = BO(
        search_space=search_space,
        obj_fun=obj_fun,
        eq_fun=h,
        model=model,
        max_FEs=10,
        DoE_size=3,
        acquisition_fun='MGFI',
        acquisition_par={'t' : 2},
        acquisition_optimization={'optimizer': 'MIES'},
        verbose=True,
        random_seed=42
    ).run()
    assert np.isclose(h(xopt), 0, atol=1e-2)

def test_BO_constraints():
    search_space = IntegerSpace([1, 10], var_name='mu') + \
        IntegerSpace([1, 10], var_name='lambda') + \
            RealSpace([0, 1], var_name='pc') + \
                RealSpace([0.005, 0.5], var_name='p')

    model = RandomForest(levels=search_space.levels)
    xopt, _, __ = BO(
        search_space=search_space,
        obj_fun=obj_fun2,
        ineq_fun=g,
        model=model,
        max_FEs=10,
        DoE_size=3,
        eval_type='dict',
        acquisition_fun='MGFI',
        acquisition_par={'t' : 2},
        n_job=1,
        n_point=1,
        verbose=True
    ).run()
    assert isinstance(xopt, dict)
    assert all(np.array(g(xopt)) <= 0)