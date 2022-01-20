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
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import polynomial_kernel
from scipy import optimize

import benchmark.bbobbenchmarks as bn


class PictureSaver:
    def __init__(self, path, extension):
        self.path = path
        self.fid = FUNCTION_ID
        self.extension = extension

    def save(self, fig, name):
        fig.savefig(self.path + name + self.fid + '.' + self.extension)


MAXX = 5
MINX = -5
DIMENSION = 2
DOESIZE = 100
OBJECTIVE_FUNCTION = bn.F21()
FUNCTION_ID = str(OBJECTIVE_FUNCTION.funId)
KPCA = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=1.1)
SAVER = PictureSaver('./', 'png')


def save_figure(fig, name):
    SAVER.save(fig, name)


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


# NO weighting scheme
def run_experiment1():
    X, Y, colours = sample_doe()

    # Original space
    fdoe = plt.figure()
    XT = get_transpose(X)
    plt.title("Original space")
    plt.scatter(XT[0], XT[1], c=colours)
    save_figure(fdoe, 'original')

    # Learning kernel pca
    KPCA.fit(X)
    X_kpca = KPCA.transform(X)
    fkpca = plt.figure()
    plt.title("Kernel PCA feature space")
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=colours)
    save_figure(fkpca, 'kpca_feature_space')

    # Learning ridge regression X -> Y
    krr = KernelRidge(kernel=KPCA.kernel,
                      kernel_params={'kernel': "rbf", 'fit_inverse_transform': True, 'gamma': 0.01})
    inverser = InverseTransformKPCA(X, krr)
    inverser.fit(X_kpca[:, 0:4])

    # Inverse of all the space
    # X_1 = inverser.inverse_all(X_kpca[:, 0:4])
    # fxinverse = plt.figure()
    # plt.title('Space after inverse transform')
    # plt.scatter(X_1[:, 0], X_1[:, 1], c=colours)
    # save_figure(fxinverse, 'original_inversed')

    # Inverse of lower dimensional manifold
    X_kpca_copy = deepcopy(X_kpca)
    for i in range(len(X_kpca_copy)):
        for j in range(1, len(X_kpca_copy[i])):
            X_kpca_copy[i][j] = 0
    X_manifold = inverser.inverse_all(X_kpca_copy[:, 0:4])
    fmankpca = plt.figure()
    plt.title("Kernel PCA manifold")
    plt.scatter(X_manifold[:, 0], X_manifold[:, 1], c='green')
    plt.scatter(XT[0], XT[1], c=colours)
    save_figure(fmankpca, 'kpca_manifold')

    plt.show()


class SystemContext:

    def __init__(self, pairwise_kernel, matrix, X, y):
        self.pairwise_kernel = pairwise_kernel
        self.matrix = matrix
        self.X = X
        self.y = y

    def k(self, x1, x2):
        return self.pairwise_kernel([x1], [x2])[0][0]

    def get_grem_line(self, x):
        grem_line = [0.] * len(self.X)
        for i in range(len(self.X)):
            grem_line[i] = self.k(x, self.X[i])
        return grem_line

    def go(self, x):
        return np.sum(np.subtract(np.matmul(self.get_grem_line(x), self.matrix), self.y) ** 2)


class InverseTransformKPCA:
    def __init__(self, X, model):
        self.X = deepcopy(X)
        self.model = model

    def fit(self, Y):
        self.model.fit(self.X, Y)
        self.W = deepcopy(self.model.dual_coef_)

    def inverse(self, y):
        global CONTEXT
        CONTEXT = SystemContext(self.model._get_kernel, self.W, self.X, y)
        initial_guess = np.array([0, 0])
        print('Initial guess fun:', f(initial_guess))
        minimized = optimize.minimize(f, x0=initial_guess, method='Nelder-Mead', bounds=[(MINX, MAXX), (MINX, MAXX)])
        print('After optimization:', minimized.fun)
        return minimized.x

    def inverse_all(self, Y):
        X1 = np.array(deepcopy(self.X))
        for (i, y) in enumerate(Y):
            X1[i] = self.inverse(y)
        return X1


CONTEXT = None


def f(x):
    return CONTEXT.go(x)


def ridge_experiment():
    r = KernelRidge(kernel=KPCA.kernel)
    X = [[0, 0], [1, 1], [0, 1], [1, 0]]
    y = [[0, 1, 0], [2, 0, 1], [1, 2, 3], [1, 1, 1]]
    r.fit(X, y)
    # r.fit([[0, 0], [1, 1], [0, 1], [1, 0]], [[0], [2], [1], [1]])
    print(r.dual_coef_)
    y1 = r.predict([[0, 2]])
    print(y1)

    winv = np.linalg.pinv(r.dual_coef_.transpose())
    print(winv)
    # print(np.matmul([2, 0, 1], winv))
    global CONTEXT
    CONTEXT = SystemContext(r._get_kernel, winv, X, [2, 0, 1])
    print(optimize.minimize(f, x0=np.array([0, 0]), method='CG', options={'gtol': 1e-7, 'maxiter': 10000}).x)


def get_inverse(krr, X, Y):
    winv = np.linalg.pinv(krr.dual_coef_.transpose())
    X_1 = np.array(deepcopy(X))
    for i, y in enumerate(Y):
        global CONTEXT
        CONTEXT = SystemContext(krr._get_kernel, winv, X, y)
        minimized = optimize.minimize(f, x0=np.array([0, 0]), method='Nelder-Mead', bounds=[(MINX, MAXX), (MINX, MAXX)])
        print(minimized.fun)
        x_1 = minimized.x
        X_1[i] = x_1
    return X_1


if __name__ == '__main__':
    random.seed(0)
    # ridge_experiment()
    run_experiment1()
