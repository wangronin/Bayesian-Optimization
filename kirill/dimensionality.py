import math
from copy import deepcopy

from sklearn.decomposition import KernelPCA, PCA

import benchmark.bbobbenchmarks as bn
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

MAXX = 5
MINX = -5
DIMENSION = 2
DOESIZE = 1000
OBJECTIVE_FUNCTION = bn.F21()
FUNCTION_ID = str(OBJECTIVE_FUNCTION.funId)
KPCA = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=1.1)


def run_experiment():
    X = []
    Y = []
    for i in range(DOESIZE):
        x = [random.uniform(MINX, MAXX) for _ in range(DIMENSION)]
        y = OBJECTIVE_FUNCTION(x)
        X.append(x)
        Y.append(y)
    fdoe = plt.figure()
    colours = compute_colours(Y)
    XT = get_transpose(X)
    plt.title("Original space")
    plt.scatter(XT[0], XT[1], c=colours)

    X_centered = get_centered_around_best(X, Y)
    X_weighted = get_rescaled_points(X_centered, Y)
    fweighted = plt.figure()
    # plt.subplot(1, 2, 1, aspect="equal")
    plt.title("Weighted DoE")
    XT_weighted = get_transpose(X_weighted)
    plt.scatter(XT_weighted[0], XT_weighted[1], c=colours)

    lpca = PCA(n_components=2)
    lpca.fit(X_weighted)
    X_lpca = lpca.transform(X)

    flpca = plt.figure()
    plt.title("Linear PCA - Feature space")
    plt.scatter(X_lpca[:, 0], X_lpca[:, 1], c=colours)

    KPCA.fit(X_weighted)
    X_kpca = KPCA.transform(X)

    fkpca = plt.figure()
    plt.title("Kernel PCA - Feature space")
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=colours)

    save_figures('/home/kirill/Projects/PhD/PlansKirill/pic/',
                 [(fdoe, 'doe'), (fweighted, 'weighted'), (flpca, 'lpca'), (fkpca, 'kpca')], FUNCTION_ID, 'pdf')
    plt.show()


def save_figures(path, figsAndNames, fid, extension):
    for (fig, name) in figsAndNames:
        fig.savefig(path + name + fid + '.' + extension)


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


def get_centered_around_best(X, y):
    X_copy = deepcopy(X)
    min_id = y.index(min(y))
    x_best = X[min_id]
    for x in X_copy:
        for ind, x_comp in enumerate(x):
            x[ind] -= x_best[ind]
    return X_copy


def get_rescaled_points(X, Y):
    w = ranking_based_weighting(Y)
    X_copy = deepcopy(X)
    # mu = compute_mean(X)
    # matrix_minus_vector(X, mu)
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
    run_experiment()
