import math
import random
import statistics
import sys
from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import KernelPCA, PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import polynomial_kernel

import benchmark.bbobbenchmarks as bn


def sample_doe(MINX, MAXX, DIMENSION, DOESIZE, OBJECTIVE_FUNCTION):
    X = []
    Y = []
    for i in range(DOESIZE):
        x = [random.uniform(MINX, MAXX) for _ in range(DIMENSION)]
        y = OBJECTIVE_FUNCTION(x)
        X.append(x)
        Y.append(y)
    colours = compute_colours_2(Y)
    return X, Y, colours


def compute_colours_2(Y):
    colours = []
    y_copy = Y.copy()
    y_copy.sort()
    min_value = y_copy[0]
    k = int(0.4 * len(Y))
    m = math.log(0.5) / (y_copy[k] - min_value)
    jet_cmap = mpl.cm.get_cmap(name='jet')
    for y in Y:
        colours.append(jet_cmap(1. - math.exp(m * (y - min_value))))
    return colours


def get_transpose(X):
    n = len(X)
    m = len(X[0])
    XT = [[0] * n for _ in range(m)]
    for i in range(n):
        for j in range(m):
            XT[j][i] = X[i][j]
    return XT


class PictureSaver:
    def __init__(self, path, function_id, extension):
        self.path = path
        self.fid = function_id
        self.extension = extension

    def save(self, fig, name):
        fig.savefig(self.path + name + self.fid + '.' + self.extension)

def get_rescaled_points(X, Y):
    w = ranking_based_weighting(Y)
    X_copy = deepcopy(X)
    mu = compute_mean(X)
    matrix_minus_vector(X, mu)
    for i in range(len(X_copy)):
        for j in range(len(X_copy[i])):
            X_copy[i][j] *= w[i]
    return X_copy


def ranking_based_weighting(Y):
    weighted_y = Y.copy()
    Y1 = [(element, ind) for ind, element in enumerate(Y)]
    Y1.sort()
    lnn = math.log(len(Y))
    for r, element in enumerate(Y1):
        _, posInX = element
        weighted_y[posInX] = lnn - math.log(r + 1)
    sum_w = sum(weighted_y)
    for ind, w in enumerate(weighted_y):
        weighted_y[ind] = w / sum_w
    return weighted_y


def compute_mean(X):
    n = len(X)
    D = len(X[0])
    sums = [0.] * D
    for i in range(n):
        for j in range(D):
            sums[j] += X[i][j]
    for ind, value in enumerate(sums):
        sums[ind] = value / n
    return sums


def matrix_minus_vector(X, vector):
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] -= vector[j]


def get_column_variances(X):
    variances = []
    for i in range(len(X[0])):
        xi = statistics.variance(X[:, i])
        variances.append(xi)
    return variances

def get_sorted_var_columns_pairs(X):
    var = get_column_variances(X)
    all_var = sum(vi for vi in var)
    var_col = [(var[i]/all_var, -i) for i in range(len(var))]
    var_col.sort()
    var_col.reverse()
    return var_col


