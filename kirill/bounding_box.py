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

# BOX_CONSTRAINTS = [[-5, 5], [-5, 5]]
# DIMENSIONS = len(BOX_CONSTRAINTS)
POINTS = 100
MAXX = 5
MINX = -5
DIMENSION = 10
DOESIZE = 1000
OBJECTIVE_FUNCTION = bn.F17()
FUNCTION_ID = str(OBJECTIVE_FUNCTION.funId) + "_16"
KPCA = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=0.5)


def sample_points(points_numer=None):
    if points_numer is None:
        points_numer = POINTS
    points = [[0. for _ in range(DIMENSIONS)] for _ in range(points_numer)]
    for p in points:
        for i in range(DIMENSIONS):
            p[i] = random.uniform(BOX_CONSTRAINTS[i][0], BOX_CONSTRAINTS[i][1])
    return points


def sample_doe(doe_size=None):
    if doe_size is None:
        doe_size = DOESIZE
    X = []
    Y = []
    for i in range(doe_size):
        x = [random.uniform(MINX, MAXX) for _ in range(DIMENSION)]
        y = OBJECTIVE_FUNCTION(x)
        X.append(x)
        Y.append(y)
    return X, Y


def k(x1, x2):
    return KPCA._get_kernel([x1], [x2])[0][0]


def f(x, fs, points):
    return k(x, x) + fs - 2. * sum(k(x, xi) for xi in points) / len(points)


def dist(a, b):
    return k(a,a) + k(b,b) - 2 * k(a,b)

def sampling_box():
    X, Y = sample_doe(100)
    # points = sample_points()
    # for p in X:
    #     points.append(p)
    # X_weighted = get_rescaled_points(X, Y)
    # KPCA.fit(X_weighted)
    # fs = 0
    # for i in range(len(points)):
    #     for j in range(len(points)):
    #         fs += k(points[i], points[j])
    # fs /= len(points) ** 2
    # r = 0
    # for x in points:
    #     r = max(r, f(x, fs, points))
    # print(r)
    
    X_weighted = get_rescaled_points(X, Y)
    KPCA.fit(X_weighted)
    N = 10
    allDist = [[0. for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            allDist[i][j] = dist(X[i], X[j])
    print(allDist)
    X_kpca = KPCA.transform(X)
    for i in range(len(X_kpca[0])):
        c = 0.
        for j in range(len(X_kpca)):
            c += X_kpca[j][i]
        # print(c/len(X_kpca))
        


def get_rescaled_points(X, Y):
    w = ranking_based_weighting(Y)
    X_copy = deepcopy(X)
    mu = compute_mean(X)
    matrix_minus_vector(X, mu)
    for i in range(len(X_copy)):
        for j in range(len(X_copy[i])):
            X_copy[i][j] *= w[i]
    return X_copy


def compute_mean(X):
    n = len(X)
    sums = [0.] * DIMENSION
    for i in range(n):
        for j in range(DIMENSION):
            sums[j] += X[i][j]
    for ind, value in enumerate(sums):
        sums[ind] = value / n
    return sums


def matrix_minus_vector(X, vector):
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] -= vector[j]


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


if __name__ == '__main__':
    random.seed(0)
    sampling_box()

