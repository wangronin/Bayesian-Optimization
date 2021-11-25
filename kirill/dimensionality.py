import math

from sklearn.decomposition import KernelPCA

import benchmark.bbobbenchmarks as bn
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

MAXX = 100
MINX = -100
DIMENSION = 2
DOESIZE = 1000


def run_experiment(objective_function):
    X = []
    Y = []
    for i in range(DOESIZE):
        x = [random.uniform(MINX, MAXX) for _ in range(DIMENSION)]
        y = objective_function(x)
        X.append(x)
        Y.append(y)
    plt.figure()
    colours = compute_colours(Y)
    XT = get_transpose(X)
    plt.title("Original space")
    plt.scatter(XT[0], XT[1], c=colours)

    center_around_best(X, Y)
    rescale_points(X, Y)
    plt.figure()
    plt.subplot(1, 2, 1, aspect="equal")
    plt.title("Weighted DoE")
    XT_reweighted = get_transpose(X)
    plt.scatter(XT_reweighted[0], XT_reweighted[1], c=colours)

    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=450)
    X_kpca = kpca.fit_transform(X)
    plt.subplot(1, 2, 2, aspect="equal")
    plt.title("Feature space")
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=colours)

    plt.show()


def compute_colours(Y):
    colours = []
    y_copy = Y.copy()
    y_copy.sort()
    min_value = y_copy[0]
    k = int(0.4 * len(Y))
    m = math.log(0.5) / (y_copy[k] - min_value)
    for y in Y:
        colours.append(get_colour(1. - math.exp(m * (y - min_value))))
    return colours


def get_colour(mix):  # linear interpolate mix=0 to mix=1
    colour0 = 'blue'
    colour1 = 'red'
    c0 = np.array(mpl.colors.to_rgb(colour0))
    c1 = np.array(mpl.colors.to_rgb(colour1))
    return mpl.colors.to_hex((1 - mix) * c0 + mix * c1)


def get_transpose(X):
    n = len(X)
    m = len(X[0])
    XT = [[0] * n for _ in range(m)]
    for i in range(n):
        for j in range(m):
            XT[j][i] = X[i][j]
    return XT


def center_around_best(X, y):
    min_id = y.index(min(y))
    x_best = X[min_id]
    for x in X:
        for ind, x_comp in enumerate(x):
            x[ind] -= x_best[ind]


def rescale_points(X, Y):
    w = ranking_based_weighting(Y)
    mu = compute_mean(X)
    matrix_minus_vector(X, mu)
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] *= w[i]


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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        random.seed(0)
    else:
        random.seed(sys.argv[1])
    run_experiment(bn.F21())
