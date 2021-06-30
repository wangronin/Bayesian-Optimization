import sys

import numpy as np

sys.path.insert(0, "./")

from bayes_optim import BO, DiscreteSpace, IntegerSpace, RealSpace
from bayes_optim.surrogate import RandomForest

seed = 666
np.random.seed(seed)
dim_r = 2  # dimension of the real values


def obj_fun0(x):
    x_r, x_i, x_d = np.array(x[:dim_r]), x[2], x[3]
    _ = 0 if x_d == "OK" else 1
    return np.sum(x_r ** 2) + abs(x_i - 10) / 123.0 + _ * 2


def obj_fun(x):
    x_r = np.array([x["continuous%d" % i] for i in range(dim_r)])
    x_i = x["ordinal"]
    x_d = x["nominal"]
    _ = 0 if x_d == "OK" else 1
    return np.sum(x_r ** 2) + abs(x_i - 10) / 123.0 + _ * 2


# Continuous variables can be specified as follows:
# a 2-D variable in [-5, 5]^2
# for 2 variables, the naming scheme is continuous0, continuous1
C = RealSpace([-5, 5], var_name="continuous") * dim_r

# Equivalently, you can also use
# C = RealSpace([[-5, 5]]] * dim)
# The general usage is:
# RealSpace([[lb_1, ub_1], [lb_2, ub_2], ..., [lb_n, ub_n]])

# Integer (ordinal) variables can be specified as follows:
# The domain of integer variables can be given as with continuous ones
# var_name is optional
I = IntegerSpace([5, 15], var_name="ordinal")

# Discrete (nominal) variables can be specified as follows:
# No lb, ub... a list of categories instead
N = DiscreteSpace(["OK", "A", "B", "C", "D", "E", "F", "G"], var_name="nominal")

# The whole search space can be constructed:
search_space = C + I + N

# Bayesian optimization also uses a Surrogate model
# For mixed variable type, the random forest is typically used
model = RandomForest(levels=search_space.levels)

opt = BO(
    search_space=search_space,
    obj_fun=obj_fun,
    model=model,
    max_FEs=50,
    DoE_size=3,  # the initial DoE size
    eval_type="dict",
    acquisition_fun="MGFI",
    acquisition_par={"t": 2},
    n_job=1,  # number of processes
    n_point=1,  # number of the candidate solution proposed in each iteration
    verbose=True,  # turn this off, if you prefer no output
)
xopt, fopt, stop_dict = opt.run()

print("xopt: {}".format(xopt))
print("fopt: {}".format(fopt))
print("stop criteria: {}".format(stop_dict))
