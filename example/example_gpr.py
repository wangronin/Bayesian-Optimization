import sys

sys.path.insert(0, "./")

import benchmark.bbobbenchmarks as bn
import numpy as np
from bayes_optim.surrogate.gaussian_process import ConstantKernel, GaussianProcess, Matern

dim = 20
f = bn.F21()

np.random.seed(42)

# X = np.random.rand(200, 20)
# y = np.array([f(x) - f.fopt for x in X])
# X = X * 10 - 5

npzfile = np.load("data.npz")
X, y = npzfile["X"], npzfile["y"]

cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
# only special if *all* dimensions are categorical
other_kernel = Matern(length_scale=np.ones(dim), length_scale_bounds=[(0.01, 100)] * dim, nu=2.5)

base_estimator = GaussianProcess(
    kernel=cov_amplitude * other_kernel,
    normalize_y=True,
    noise="gaussian",
    n_restarts_optimizer=2,
    random_state=np.random.randint(1000),
    lb=np.array([0] * 20),
    ub=np.array([1] * 20),
)
base_estimator.fit(X, y)
mu, std, mu_grad, std_grad = base_estimator.predict(
    X[0:1], return_std=True, return_mean_grad=True, return_std_grad=True
)
print(mu)
print(std)
print(mu_grad)
print(std_grad)
