# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:19:28 2015

@author: wangronin
"""

import pdb
import numpy as np
from numpy import pi, log

## SMSE measurement
# test_y is the target, pred_y the predicted target, both 1D arrays of same length
def SMSE(test_y, pred_y):
    se = []
    target_variance = np.var(test_y)
    for i in range(len(test_y)):
        temp = (pred_y[i] - test_y[i]) ** 2
        se.append(temp)
    mse = np.mean(se)
    smse = mse / target_variance
    return smse


## MSLL = mean standardized log loss
## logprob = 0.5*log(2*pi.*varsigmaVec) + sserror - 0.5*log(2*pi*varyTrain)...
##           - ((yTestVec - meanyTrain).^2)./(2*varyTrain);
def MSLL(train_y, test_y, pred_y, variances):
    sll = []
    mean_y = np.mean(train_y)
    var_y = np.var(train_y)
    for i in range(len(variances)):
        if variances[i] == 0:
            variances[i] += 0.0000001  # hack
        sll_trivial = 0.5 * log(2 * pi * var_y) + ((test_y[i] - mean_y) ** 2 / (2 * var_y))
        sllv = (
            0.5 * log(2 * pi * variances[i]) + ((test_y[i] - pred_y[i]) ** 2 / (2 * variances[i]))
        ) - sll_trivial
        sll.append(sllv)
    sll = np.array(sll)
    msll = np.mean(sll)
    return msll


# # Obtain the initial design locations
# def get_design_sites(dim, n_sample, x_lb, x_ub, sampling_method='lhs'):

#     x_lb = atleast_2d(x_lb)
#     x_ub = atleast_2d(x_ub)

#     x_lb = x_lb.T if size(x_lb, 0) != 1 else x_lb
#     x_ub = x_ub.T if size(x_ub, 0) != 1 else x_ub

#     if sampling_method == 'lhs':
#         # Latin Hyper Cube Sampling: Get evenly distributed sampling in R^dim
#         samples = lhs(dim, samples=n_sample) * (x_ub - x_lb) + x_lb

#     elif sampling_method == 'uniform':
#         samples = np.random.rand(n_sample, dim) * (x_ub - x_lb) + x_lb

#     elif sampling_method == 'sobol':
#         seed = mod(int(time.time()) + os.getpid(), int(1e6))
#         samples = np.zeros((n_sample, dim))
#         for i in range(n_sample):
#             samples[i, :], seed = i4_sobol(dim, seed)
#         samples = samples * (x_ub - x_lb) + x_lb

#     elif sampling_method == 'halton':
#         sequencer = Halton(dim)
#         samples = sequencer.get(n_sample) * (x_ub - x_lb) + x_lb

#     return samples

# diagostic plots of gradient field
# def plot_contour_gradient(ax, f, grad, x_lb, x_ub, title='f', is_log=False, n_level=30, foo=0,
#                          f_data=None, grad_data=None, n_per_axis=200):
#    fig = ax.figure

#    x = np.linspace(x_lb[0], x_ub[0], n_per_axis)
#    y = np.linspace(x_lb[1], x_ub[1], n_per_axis)
#    X, Y = np.meshgrid(x, y)

# #    divider = make_axes_locatable(ax)
# #    cax = divider.append_axes("right", size="5%", pad=0.05)

#    if f_data is None:
#        fitness = np.array([f(p.reshape(1, -1)) for p in np.c_[X.flatten(), Y.flatten()]]).reshape(-1, len(x))
#    else:
#        fitness = f_data
# #       fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness)) + foo
#    if is_log:
#        fitness = np.log(fitness)

#    CS = ax.contour(X, Y, fitness, n_level, cmap=plt.cm.Spectral, linewidths=1)
#    plt.clabel(CS, inline=1, fontsize=10)

#    if grad is not None:
#        # calculate function gradients
#        x1 = np.linspace(x_lb[0], x_ub[0], np.floor(n_per_axis / 10))
#        x2 = np.linspace(x_lb[1], x_ub[1], np.floor(n_per_axis / 10))
#        X1, X2 = np.meshgrid(x1, x2)
#        if grad_data is None:
#            dx = np.array([grad(p.reshape(1, -1)).flatten() for p in np.c_[X1.flatten(), X2.flatten()]])
#            np.save('grad.npy', dx)
#        else:
#            dx = grad_data

#        dx_norm = np.sqrt(np.sum(dx ** 2.0, axis=1)) # in case of zero gradients
#        dx /= dx_norm.reshape(-1, 1)
#        dx1 = dx[:, 0].reshape(-1, len(x1))
#        dx2 = dx[:, 1].reshape(-1, len(x1))

#        CS = ax.quiver(X1, X2, dx1, dx2, dx_norm, cmap=plt.cm.jet,
#                       #norm=colors.LogNorm(vmin=1e-100, vmax=dx_norm.max()),
#                       headlength=5)

# #        fig.colorbar(CS, ax=ax)

#    ax.set_xlabel('$x_1$')
#    ax.set_ylabel('$x_2$')
#    ax.grid(True)
#    ax.set_title(title)
#    ax.set_xlim(x_lb[0], x_ub[0])
#    ax.set_ylim(x_lb[1], x_ub[1])


# def plot_surface_contour(ax, f, grad, x_lb, x_ub, title='f',
#                          log_transform=False, n_level=30, foo=0,
#                          f_data=None, grad_data=None,n_per_axis=200):

#     fig = ax.figure
#     x = np.linspace(x_lb[0], x_ub[0], n_per_axis)
#     y = np.linspace(x_lb[1], x_ub[1], n_per_axis)
#     X, Y = np.meshgrid(x, y)

# #    divider = make_axes_locatable(ax)
# #    cax = divider.append_axes("right", size="5%", pad=0.05)

#     if f_data is None:
#         fitness = np.array([f(p.reshape(1, -1)) for p in np.c_[X.flatten(), Y.flatten()]]).reshape(-1, len(x))
#     else:
#         fitness = f_data
#     try:
#         fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness)) + foo
#         if log_transform:
#             fitness = np.log(fitness)
#         CS = ax.contour(X, Y, fitness, n_level, cmap=plt.cm.winter, linewidths=1, offset=0)
#         plt.clabel(CS, inline=1, fontsize=15)

#         tri = mtri.Triangulation(X, Y)
#         ax.plot_trisurf(X, Y, fitness, rstride=1, cstride=1, cmap=plt.cm.Spectral, linewidth=0, alpha=0.3)
# #        fig.colorbar(CS, ax=ax, fraction=0.046, pad=0.04)
#     except:
#         pdb.set_trace()

#    # calculate function gradients
#     x1 = np.linspace(x_lb[0], x_ub[0], np.floor(n_per_axis / 10))
#     x2 = np.linspace(x_lb[1], x_ub[1], np.floor(n_per_axis / 10))
#     X1, X2 = np.meshgrid(x1, x2)

#     if grad is not None:
#         if grad_data is None:
#             dx = np.array([grad(p).flatten() for p in np.c_[X1.flatten(), X2.flatten()]])
#             np.save('grad.npy', dx)
#         else:
#             dx = grad_data

#         dx_norm = np.sqrt(np.sum(dx ** 2.0, axis=1)) # in case of zero gradients
#         dx /= dx_norm.reshape(-1, 1)
#         dx1 = dx[:, 0].reshape(-1, len(x1))
#         dx2 = dx[:, 1].reshape(-1, len(x1))

#         CS = ax.quiver(X1, X2, dx1, dx2, dx_norm, cmap=plt.cm.jet,
#                       #norm=colors.LogNorm(vmin=1e-100, vmax=dx_norm.max()),
#                       headlength=5)

# #        fig.colorbar(CS, ax=ax)

#     ax.set_xlabel('$x_1$')
#     ax.set_ylabel('$x_2$')
#     ax.grid(True)
#     ax.set_title(title)
#     ax.set_xlim(x_lb[0], x_ub[0])
#     ax.set_ylim(x_lb[1], x_ub[1])
