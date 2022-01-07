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

import benchmark.bbobbenchmarks as bn

MAXX = 5
MINX = -5
DIMENSION = 2
DOESIZE = 1000
OBJECTIVE_FUNCTION = bn.F17()
FUNCTION_ID = str(OBJECTIVE_FUNCTION.funId) + "_16"
KPCA = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=1)


def run_experiment():
    X, Y, colours = sample_doe()
    fdoe = plt.figure()
    XT = get_transpose(X)
    plt.title("Original space")
    plt.scatter(XT[0], XT[1], c=colours)

    # X_centered = get_centered_around_best(X, Y)
    X_c, X_c_inverse = get_rescaled_points_cca(X, Y)
    fcca = plt.figure()
    plt.title("CCA")
    plt.scatter(X_c[:, 0], X_c[:, 1], c=colours)

    X_weighted = get_rescaled_points(X, Y)
    # X_weighted = X_c
    fweighted = plt.figure()
    # plt.subplot(1, 2, 1, aspect="equal")
    plt.title("Weighted DoE")
    XT_weighted = get_transpose(X_weighted)
    plt.scatter(XT_weighted[0], XT_weighted[1], c=colours)

    lpca = PCA(n_components=2)
    lpca.fit(X_weighted)
    X_lpca = lpca.transform(X)

    X_lpca_inverse = get_main_component_inverse_transform(X_lpca, lpca)
    inverselpca = plt.figure()
    plt.title("Linear PCA - Inverse of points")
    plt.plot(X_lpca_inverse[:, 0], X_lpca_inverse[:, 1], c='green')
    plt.scatter(XT[0], XT[1], c=colours)

    # X_lpca_inverse = get_main_component_inverse_transform(X_lpca, lpca)
    inverseCCA = plt.figure()
    plt.title("CCA - Inverse of points")
    plt.plot(X_c_inverse[:, 0], X_c_inverse[:, 1], c='green')
    plt.scatter(XT[0], XT[1], c=colours)

    flpca = plt.figure()
    plt.title("Linear PCA - Feature space")
    plt.scatter(X_lpca[:, 0], X_lpca[:, 1], c=colours)

    KPCA.fit(X_weighted)
    X_kpca = KPCA.transform(X)

    fkpca = plt.figure()
    plt.title("Kernel PCA - Feature space")
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=colours)

    krr = KernelRidge()
    krr.fit(X_kpca[:, 0:2], X)

    X_inverse = krr.predict(X_kpca[:, 0:2])
    inverseAllkpca = plt.figure()
    plt.title("Kernel PCA - Inverse")
    plt.scatter(X_inverse[:, 0], X_inverse[:, 1], c=colours)

    X_kpca_inverse = get_main_component_approx_inverse(X_kpca, krr)
    inversekpca = plt.figure()
    plt.title("Kernel PCA - Inverse of points")
    plt.plot(X_kpca_inverse[:, 0], X_kpca_inverse[:, 1], c='green')
    plt.scatter(XT[0], XT[1], c=colours)

    save_figures('/home/kirill/Projects/PhD/PlansKirill/pic/',
                 [(fdoe, 'doe'), (fweighted, 'weighted'), (flpca, 'lpca'), (fkpca, 'kpca'),
                  (inverselpca, 'inverseLpca'), (inversekpca, 'inverseKpca'), (inverseAllkpca, 'inverseAllKpca'),
                  (fcca, 'CCA'), (inverseCCA, 'inverseCCA')],
                 FUNCTION_ID, 'pdf')
    plt.show()


def sorted_variance_experiment():
    X, Y, colours = sample_doe()
    X_weighted = get_rescaled_points(X, Y)
    KPCA.fit(X_weighted)
    X_kpca = KPCA.transform(X)
    vars = get_colum_variances(X_kpca)
    all_var = sum(vi ** 2 for vi in vars)
    var = []
    for vi in vars:
        var.append(vi ** 2 / all_var)
    var.sort()
    var.reverse()
    bar = plt.figure()
    plt.bar(np.arange(len(var)), var)
    plt.ylabel("$ {\sigma^2_i}/{\sum \sigma^2_i}$")
    plt.xlabel("$\sigma^2_i$")
    plt.title("Sorted variances bar chart, kernel = " + KPCA.kernel + ", $\gamma$ = " + str(KPCA.gamma))
    plt.show()
    save_figures('/home/kirill/Projects/PhD/PlansKirill/pic/',
                 [(bar, 'variances_bar')],
                 FUNCTION_ID, 'pdf')


def get_colum_variances(X):
    variances = []
    for i in range(len(X[0])):
        xi = statistics.variance(X[:, i])
        variances.append(xi)
    return variances


def run_experiment_1():
    # 1. Original colored sample set
    X, Y, colours = sample_doe()
    XT = get_transpose(X)
    fdoe = plt.figure()
    plt.title("Original colored sample set")
    plt.scatter(XT[0], XT[1], c=colours)

    # 2. Weighted sample matrix around the mean
    X_weighted = get_rescaled_points(X, Y)
    fweighted = plt.figure()
    plt.title("Weighted sample matrix around the mean")
    XT_weighted = get_transpose(X_weighted)
    plt.scatter(XT_weighted[0], XT_weighted[1], c=colours)

    # 3. Linear PCA feature space
    lpca = PCA(n_components=2)
    lpca.fit(X_weighted)
    X_lpca = lpca.transform(X)
    lpca_feature_space = plt.figure()
    plt.title("Linear PCA feature space")
    plt.scatter(X_lpca[:, 0], X_lpca[:, 1], c=colours)

    # 4. Kernel PCA feature space
    KPCA.fit(X_weighted)
    X_kpca = KPCA.transform(X)
    fkpca = plt.figure()
    plt.title("Kernel PCA feature space")
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=colours)

    # 5. Inverse transformation of new sample matrix with linear PCA
    X_lpca_inverse = lpca.inverse_transform(X_lpca)
    lpca_inverse = plt.figure()
    plt.title("Inverse transformation of new sample matrix with linear PCA")
    plt.scatter(X_lpca_inverse[:, 0], X_lpca_inverse[:, 1], c=colours)

    # 6. Inverse transformation of new sample matrix with linear CCA
    X_c, X_c_inverse, X_c_main_inverse = get_rescaled_points_cca(X, Y)
    cca_inverse = plt.figure()
    plt.title("Inverse transformation of new sample matrix with linear CCA")
    plt.scatter(X_c_inverse[:, 0], X_c_inverse[:, 1], c=colours)

    # 7. Inverse transformation of first component (lower-dimensional manifold) plotted on the original sample set with linear PCA
    X_lpca_main_inverse = get_main_component_inverse_transform(X_lpca, lpca)
    lower_manifold_lpca = plt.figure()
    plt.title("Lower-dimensional manifold with linear PCA")
    plt.plot(X_lpca_main_inverse[:, 0], X_lpca_main_inverse[:, 1], c='green')
    plt.scatter(XT[0], XT[1], c=colours)

    # 8. Inverse transformation of first component (lower-dimensional manifold) plotted on the original sample set with linear CCA
    lower_manifold_cca = plt.figure()
    plt.title("Lower-dimensional manifold with linear CCA")
    plt.plot(X_c_main_inverse[:, 0], X_c_main_inverse[:, 1], c='green')
    plt.scatter(XT[0], XT[1], c=colours)

    save_figures('/home/kirill/Projects/PhD/PlansKirill/pic/',
                 [(fdoe, 'doe'),
                  (fweighted, 'weighted'),
                  (lpca_feature_space, 'pca_feature_space'),
                  (fkpca, 'kpca_feature_space'),
                  (lpca_inverse, 'lpca_inverse'),
                  (cca_inverse, 'cca_inverse'),
                  (lower_manifold_lpca, 'lower_mainfold_lpca'),
                  (lower_manifold_cca, 'lower_manifold_cca')],
                 FUNCTION_ID, 'pdf')
    plt.show()


def inverse_transform_of_main(num):
    X, Y, colours = sample_doe()
    XT = get_transpose(X)
    fdoe = plt.figure()
    plt.title("Original colored sample set")
    plt.scatter(XT[0], XT[1], c=colours)
    lpca = PCA(n_components=2)
    X_weighted = get_rescaled_points(X, Y)
    lpca.fit(X_weighted)
    X_lpca = lpca.transform(X)
    X_lpca_main_inverse = get_main_component_inverse_transform(X_lpca, lpca)
    lower_manifold_lpca = plt.figure()
    plt.title("Lower-dimensional manifold with linear PCA")
    plt.plot(X_lpca_main_inverse[:, 0], X_lpca_main_inverse[:, 1], c='green')
    plt.scatter(XT[0], XT[1], c=colours)
    plt.ylabel(str(num), loc='top')
    save_figures('/home/kirill/Projects/PhD/PlansKirill/pic/',
                 [(fdoe, 'doe'),
                  (lower_manifold_lpca, 'lower_mainfold_lpca_arg' + str(num) + '_')],
                 FUNCTION_ID, 'pdf')


def inverse_transform_of_main_2():
    for doe_size in [1000, 500, 100, 50]:
        lower_manifold_lpca = plt.figure()
        for num in range(1, 101):
            random.seed(a=None)
            X, Y, colours = sample_doe(doe_size)
            lpca = PCA(n_components=2)
            X_weighted = get_rescaled_points(X, Y)
            lpca.fit(X_weighted)
            X_lpca = lpca.transform(X)
            X_lpca_main_inverse = get_main_component_inverse_transform(X_lpca, lpca)
            plt.title("Lower-dimensional manifold with linear PCA, doe size is " + str(doe_size))
            plt.plot(X_lpca_main_inverse[:, 0], X_lpca_main_inverse[:, 1], c=(num / 100, 0.5, 0.5), label=str(num))
            # plt.legend()
        save_figures('/home/kirill/Projects/PhD/PlansKirill/pic/',
                     [(lower_manifold_lpca, 'lower_mainfold_lpca_doe_' + str(doe_size) + '_')],
                     FUNCTION_ID, 'pdf')


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
    colours = compute_colours_2(Y)
    return X, Y, colours


def get_main_component_inverse_transform(X, pca):
    X_copy = deepcopy(X)
    for i in range(len(X_copy)):
        for j in range(1, len(X_copy[i])):
            X_copy[i][j] = 0
    return pca.inverse_transform(X_copy)


def get_main_component_approx_inverse(X, model):
    X_copy = deepcopy(X)
    for i in range(len(X_copy)):
        for j in range(1, len(X_copy[i])):
            X_copy[i][j] = 0
    return model.predict(X_copy[:, 0:2])


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
    mu = compute_mean(X)
    matrix_minus_vector(X, mu)
    for i in range(len(X_copy)):
        for j in range(len(X_copy[i])):
            X_copy[i][j] *= w[i]
    return X_copy


def get_rescaled_points_cca(X, Y):
    Y_duplicated = [[Y[i], Y[i]] for i in range(len(X))]
    cca = CCA(n_components=2)
    cca.fit(X, Y_duplicated)
    X_c = cca.transform(X)
    X_c_inverse = cca.inverse_transform(X_c)
    X_c_main_inverse = get_main_component_inverse_transform(X_c, cca)
    return X_c, X_c_inverse, X_c_main_inverse


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
        sorted_variance_experiment()
    else:
        # random.seed(sys.argv[1])
        inverse_transform_of_main_2()
    # run_experiment_1()
