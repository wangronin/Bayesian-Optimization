# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:57:13 2017

@author: Hao Wang
@email: wangronin@gmail.com
"""

import pdb

from GaussianProcess.trend import constant_trend, linear_trend, quadratic_trend
from GaussianProcess import GaussianProcess as GaussianProcess
from GaussianProcess.utils import plot_contour_gradient

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import r2_score
from InfillCriteria import EI

import matplotlib.pyplot as plt
from matplotlib import rcParams
#from mpl_toolkits.mplot3d import axes3d

from deap import benchmarks

import numpy as np
from numpy.random import randn

np.random.seed(666)

plt.ioff()
plt.style.use('ggplot')
rcParams['legend.numpoints'] = 1
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['xtick.major.size'] = 10
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.size'] = 10
rcParams['ytick.major.width'] = 1
rcParams['axes.labelsize'] = 13
rcParams['font.size'] = 13
rcParams['lines.markersize'] = 11
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'

fig_width = 10
fig_height = fig_width / 1.1

dim = 2
n_init_sample = 20
noise_var = 0

def fitness(X):
    # x1, x2 = X[:, 0], X[:, 1]
    # a, b, c, r, s, t = 1, 5.1 / (4*pi**2), 5/pi, 6., 10., 1 / (8*pi)
    # return a * (x2 - b*x1 ** 2. + c*x1 - r) ** 2. + s*(1-t)*cos(x1) + s
    # y = np.sum(X ** 2., axis=1) + 1e-1 *  np.random.randn(X.shape[0])
    # return y
    X = np.atleast_2d(X)
    return np.array([benchmarks.sphere(x)[0] for x in X]) \
        + np.sqrt(noise_var) * randn(X.shape[0])


x_lb = np.array([-5] * dim)
x_ub = np.array([5] * dim)

X = np.random.rand(n_init_sample, dim) * (x_ub - x_lb) + x_lb
y = fitness(X)

length_lb = [1e-10] * dim
length_ub = [1e3] * dim
# thetaL = length_ub ** -2.  / (x_ub - x_lb) ** 2. * np.ones(dim)
# thetaU = length_lb ** -2.  / (x_ub - x_lb) ** 2. * np.ones(dim)

thetaL = 1e-5 * (x_ub - x_lb) * np.ones(dim)
thetaU = 10 * (x_ub - x_lb) * np.ones(dim)

# initial search point for hyper-parameters
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

if 1 < 2:
    mean = constant_trend(dim, beta=0)
    model = GaussianProcess(mean=mean,
                            corr='matern',
                            theta0=theta0,
                            thetaL=thetaL,
                            thetaU=thetaU,
                            nugget=0,
                            noise_estim=False,
                            optimizer='BFGS',
                            verbose=True,
                            wait_iter=5,
                            random_start=30,
                            likelihood='concentrated',
                            eval_budget=200)

    # from GaussianProcess import OWCK
    # model = OWCK(corr='matern',
    #              n_cluster=5,
    #              min_leaf_size=50,
    #              cluster_method='k-mean',
    #              overlap=0.0,
    #              verbose=True,
    #              theta0=theta0,
    #              thetaL=thetaL,
    #              thetaU=thetaU,
    #              nugget=1e-10,
    #              random_start=30,
    #              optimizer='BFGS',
    #              nugget_estim=False,
    #              normalize=False,
    #              is_parallel=False)

else:
    length0 = 1. / np.sqrt(theta0)
    length_lb = [1e-10] * dim
    length_ub = [1e2] * dim
    kernel = 1.0 * Matern(length_scale=(1, 1), length_scale_bounds=(1e-10, 1e2))
    model = GaussianProcessRegressor(kernel, alpha=0, n_restarts_optimizer=30, normalize_y=False)

model.fit(X, y)

X_test = np.random.rand(int(1e4), dim) * (x_ub - x_lb) + x_lb
y_test = fitness(X_test)

y_hat = model.predict(X_test)
r2 = r2_score(y_test, y_hat)

f = lambda x: model.predict(x)
# sd2 = lambda x: model.predict(x, return_std=True)[1]
sd2 = lambda x: model.predict(x, eval_MSE=True)[1]

# ei = EI(model)
# ei_dx = lambda x: ei(x, dx=True)[1]
ei_dx = None
ei = f

# f_dx = lambda x: model.gradient(x)[0]
# sd2_dx = lambda x: model.gradient(x)[1]

f_dx = None
sd2_dx = None

print()
print('R2:', r2)
print('Homoscedastic noise variance:', noise_var)
# print('noise variance learned:', model.noise_var)
print()
print('Parameter optimization')
print('initial guess:', theta0)
# print('optimum:', model.theta_)

#fig0, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True, sharex=False,
#                                figsize=(fig_width, fig_height),
#                                subplot_kw={'aspect': 'equal'}, dpi=100)
fig0, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharey=True, sharex=True,
                                figsize=(fig_width, fig_height), dpi=100,
                                subplot_kw={'aspect': 'equal'})

plot_contour_gradient(ax0, fitness, None, x_lb, x_ub, title='Target function',
                      n_level=15, n_per_axis=200)

plot_contour_gradient(ax1, ei, ei_dx, x_lb, x_ub, title='log(Expected Improvement)', is_log=True,
                      n_level=15, n_per_axis=150)

plot_contour_gradient(ax2, f, f_dx, x_lb, x_ub, title='GPR prediction',
                      n_level=15, n_per_axis=150)
                      
plot_contour_gradient(ax3, sd2, sd2_dx, x_lb, x_ub, title='log(GPR MSE)', is_log=True,
                      n_level=15, n_per_axis=150)

ax0.plot(X[:, 0], X[:, 1], ls='none', marker='.',
         ms=15, mfc='k', mec='none', alpha=0.8)

ax0.set_ylabel('$x_2$')
ax2.set_ylabel('$x_2$')
ax2.set_xlabel('$x_1$')
ax3.set_xlabel('$x_1$')

for ax in (ax0, ax1, ax2, ax3):
    ax.set_xlim(x_lb[0], x_ub[0])
    ax.set_ylim(x_lb[1], x_ub[1])

plt.tight_layout()
plt.show()
