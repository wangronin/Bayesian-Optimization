# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:15:50 2017

@author: wangronin
"""

import pdb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


# diagostic plots of gradient field 
def plot_contour_gradient(ax, f, grad, x_lb, x_ub, title='f', is_log=False, n_level=30, foo=0,
                          f_data=None, grad_data=None,n_per_axis=200):
    
    fig = ax.figure
    
    x = np.linspace(x_lb[0], x_ub[0], n_per_axis)
    y = np.linspace(x_lb[1], x_ub[1], n_per_axis) 
    X, Y = np.meshgrid(x, y)
    
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    if f_data is None:
        fitness = np.array([f(p) for p in np.c_[X.flatten(), Y.flatten()]]).reshape(-1, len(x))
    else:
        fitness = f_data
    try:
        fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness)) + foo
        if is_log:
            fitness = np.log(fitness)
        CS = ax.contour(X, Y, fitness, n_level, cmap=plt.cm.winter, linewidths=1)
#        plt.clabel(CS, inline=1, fontsize=5)
#        fig.colorbar(CS, ax=ax, fraction=0.046, pad=0.04)
    except:
        pdb.set_trace()
    
    # calculate function gradients   
    x1 = np.linspace(x_lb[0], x_ub[0], np.floor(n_per_axis / 10))
    x2 = np.linspace(x_lb[1], x_ub[1], np.floor(n_per_axis / 10)) 
    X1, X2 = np.meshgrid(x1, x2)     
    
    if grad is not None:
        if grad_data is None:
            dx = np.array([grad(p).flatten() for p in np.c_[X1.flatten(), X2.flatten()]])
            np.save('grad.npy', dx)
        else:
            dx = grad_data
          
        dx_norm = np.sqrt(np.sum(dx ** 2.0, axis=1)) # in case of zero gradients
        dx /= dx_norm.reshape(-1, 1)
        dx1 = dx[:, 0].reshape(-1, len(x1))
        dx2 = dx[:, 1].reshape(-1, len(x1))
      
        CS = ax.quiver(X1, X2, dx1, dx2, dx_norm, cmap=plt.cm.jet, 
                       #norm=colors.LogNorm(vmin=1e-100, vmax=dx_norm.max()),
                       headlength=5)
   
#        fig.colorbar(CS, ax=ax)
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlim(x_lb[0], x_ub[0])
    ax.set_ylim(x_lb[1], x_ub[1])