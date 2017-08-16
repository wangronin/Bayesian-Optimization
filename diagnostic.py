# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:57:13 2017

@author: wangronin
"""

import pdb

from GaussianProcess import OWCK
from GaussianProcess import GaussianProcess_extra as GaussianProcess
from GaussianProcess.utils import plot_contour_gradient

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import r2_score

#import matplotlib.pyplot as plt
from deap import benchmarks

import numpy as np
from numpy import pi, cos

np.random.seed(100)

plt.ioff()
fig_width = 20
fig_height = fig_width / 3.2

def fitness(X):

    X = np.atleast_2d(X)
#    x1, x2 = X[:, 0], X[:, 1]
#    a, b, c, r, s, t = 1, 5.1 / (4*pi**2), 5/pi, 6., 10., 1 / (8*pi)
#    return a * (x2 - b*x1 ** 2. + c*x1 - r) ** 2. + s*(1-t)*cos(x1) + s
#    y = np.sum(X ** 2., axis=1) + 1e-1 *  np.random.randn(X.shape[0])
#    return y

    return np.array([benchmarks.ackley(x)[0] for x in X])

dim = 2
alpha = np.pi / 6.
n_init_sample = 500

x_lb = np.array([-5] * dim)
x_ub = np.array([5] * dim)

X = np.random.rand(n_init_sample, dim) * (x_ub - x_lb) + x_lb
y = fitness(X)

length_lb = 1e-10
length_ub = 1e2
#thetaL = length_ub ** -2.  / (x_ub - x_lb) ** 2. * np.ones(dim)
#thetaU = length_lb ** -2.  / (x_ub - x_lb) ** 2. * np.ones(dim)

thetaL = 1e-5 * (x_ub - x_lb) * np.ones(dim)
thetaU = 10 * (x_ub - x_lb) * np.ones(dim)

# initial search point for hyper-parameters
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

if 1 < 2:
#    model = GaussianProcess(corr='matern',
##                            theta0=np.array([0.86637243,  0.78871857]) ** -2,
#                            theta0 = theta0,
#                            thetaL=thetaL,
#                            thetaU=thetaU,
#                            nugget=1e-10,
#                            nugget_estim=True,
#                            optimizer='BFGS',
#                            verbose=True,
#                            wait_iter=10,
#                            random_start=30,
#                            normalize=False)

    model = OWCK(corr='matern',
                 n_cluster=5,
                 min_leaf_size=50,
                 cluster_method='k-mean',
                 overlap=0.0,
                 verbose=True,
                 theta0=theta0,
                 thetaL=thetaL,
                 thetaU=thetaU,
                 nugget=1e-10,
                 random_start=30,
                 optimizer='BFGS',
                 nugget_estim=False,
                 normalize=False,
                 is_parallel=False)

else:
    l = 1 / np.sqrt(theta0)
    kernel = 1.0 * Matern(length_scale=l, length_scale_bounds=(length_lb, length_ub))
    model = GaussianProcessRegressor(kernel, alpha=1e-10, n_restarts_optimizer=20,
                                     normalize_y = True)

model.fit(X, y)

y_hat = model.predict(X)
r2 = r2_score(y, y_hat)
print r2

#print model.noise_var

fig0, (ax0, ax1) = plt.subplots(1, 2, sharey=True, sharex=False,
                                     figsize=(fig_width, fig_height),
                                     subplot_kw={'aspect':'equal'}, dpi=100)

f = lambda x: model.predict(x)

plot_contour_gradient(ax0, fitness, None, x_lb, x_ub, title='Function',
                      is_log=True, n_level=15, foo=1, n_per_axis=100)

plot_contour_gradient(ax1, f, None, x_lb, x_ub, title='Kriging model',
                      n_level=15, n_per_axis=100)


ax0.plot(X[:, 0], X[:, 1], ls='none', marker='.', ms=10, mfc='k', mec='none',
         alpha=0.7)

for i, ax in enumerate((ax0, ax1)):
    ax.set_xlim(x_lb[i], x_ub[i])
    ax.set_xlim(x_lb[i], x_ub[i])

plt.tight_layout()
plt.show()

pdb.set_trace()
