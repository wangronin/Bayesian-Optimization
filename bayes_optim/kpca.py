import random
from copy import deepcopy
from functools import partial
from math import exp
from sys import breakpointhook

import numpy as np
from scipy import optimize
from scipy.spatial.distance import cdist

from .mylogging import *


def additive_chi2_kernel(a, b):
    # TODO
    raise NotImplementedError


def chi2_kernel(a, b):
    # TODO
    raise NotImplementedError


def linear_kernel(a, b):
    # TODO
    raise NotImplementedError


def polynomial_kernel(a, b, gamma, d, c0, **kwargs):
    return (sum(ai * bi * gamma for ai, bi in zip(a, b)) + c0) ** d


def rbf_kernel(a, b, gamma, **kwargs):
    return exp(-gamma * np.sum((np.array(a) - np.array(b)) ** 2))


def laplacian_kernel(a, b):
    # TODO
    raise NotImplementedError


def sigmoid_kernel(a, b):
    # TODO
    raise NotImplementedError


def cosine_similarity(a, b):
    # TODO
    raise NotImplementedError


PAIRWISE_KERNEL_FUNCTIONS = {
    "additive_chi2": additive_chi2_kernel,
    "chi2": chi2_kernel,
    "linear": linear_kernel,
    "polynomial": polynomial_kernel,
    "poly": polynomial_kernel,
    "rbf": rbf_kernel,
    "laplacian": laplacian_kernel,
    "sigmoid": sigmoid_kernel,
    "cosine": cosine_similarity,
}


def create_kernel(kernel_name, parameters):
    if kernel_name == "__internal_rbf":
        return partial(__internal_rbf, **parameters)
    kernel_function = PAIRWISE_KERNEL_FUNCTIONS[kernel_name]
    return partial(kernel_function, **parameters)


class MyKernelPCA:
    # Implementation is based on paper García_González_et_al_2021_A_kernel_Principal_Component_Analysis
    def __init__(self, epsilon, kernel_config, dimensions: int = None, NN: int = None):
        self.kernel_config = kernel_config
        self.epsilon = epsilon
        self.NN = dimensions if NN is None else NN
        self._reuse_rbf_data = False

    def enable_inverse_transform(self, bounds):
        self.bounds = bounds

    def set_initial_space_points(self, X):
        self.X = np.array(X.tolist())

    @staticmethod
    def l2(x):
        ans = 0
        for i in range(len(x)):
            ans += x[i] ** 2
        return ans

    @staticmethod
    def hparam_cost_func(conical_w, X, good_subspace, kernel, z_star, G, V, bounds):
        ns = len(X)
        x_ = MyKernelPCA.linear_combination(conical_w, good_subspace)
        g_star = np.array([kernel(X[i], x_) for i in range(ns)])
        z = V @ MyKernelPCA.__center_g(g_star, G)
        bounds_ = np.atleast_2d(bounds)
        idx_lower = np.where(x_ < bounds_[:, 0])[0]
        idx_upper = np.where(x_ > bounds_[:, 1])[0]
        penalty = np.sum([bounds_[i, 0] - x_[i] for i in idx_lower]) + np.sum(
            [x_[i] - bounds_[i, 1] for i in idx_upper]
        )
        return np.log10(np.sum((z_star.ravel() - z) ** 2)) * 5 + penalty

    @staticmethod
    def linear_combination(w, X):
        comb = [0.0] * len(X[0])
        for i in range(len(X)):
            for j in range(len(X[0])):
                comb[j] += w[i] * X[i][j]
        return comb

    def _set_reuse_rbf_data(self, l2_norm_matrix):
        kernel_name = self.kernel_config["kernel_name"]
        if kernel_name != "rbf":
            raise ValueError(f"rbf kernel is expected, but {kernel_name} found")
        self._reuse_rbf_data = True
        self._l2_norm_matrix = l2_norm_matrix

    def __sorted_eig(self, X: np.ndarray):
        values, vectors = np.linalg.eigh(X)
        values = values.view(np.float64)
        vectors = vectors.view(np.float64)
        if any(~np.isclose(values[values < 0], 0)):  # NOTE: this should not happen
            breakpoint()
        values[values < 0] = 0
        return values, vectors

    def __compute_G(self, X) -> np.ndarray:
        gamma = self.kernel_config["kernel_parameters"]["gamma"]
        D = self._l2_norm_matrix if self._reuse_rbf_data else cdist(X, X) ** 2
        return np.exp(-gamma * D)

    def __center_G(self, G: np.ndarray) -> np.ndarray:
        ns = len(G)
        O = np.ones((ns, ns))
        M = G @ O
        return G - M / ns - M.T / ns + M.T.dot(O) / ns**2

    @staticmethod
    def __center_g(g: np.ndarray, G: np.ndarray) -> np.ndarray:
        n = len(g)
        g = g.reshape(-1, 1)
        t = G @ np.ones((n, 1))
        return g - t / n - np.ones((n, n)) @ g / n + np.tile(np.sum(t) / n**2, (n, 1))

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.kernel = create_kernel(
            self.kernel_config["kernel_name"], self.kernel_config["kernel_parameters"]
        )
        self.X = X
        ns = len(X)
        y_ = ((y - np.mean(y)) / np.std(y)).reshape(-1, 1)
        # compute the raw Gram matrix
        G = self.__compute_G(self.X)
        # center and weight the Gram matrix
        G_centered = self.__center_G(G)
        self.G = G
        self.too_compressed = False
        # eigendecomposition
        eigen_values, eigen_vectors = self.__sorted_eig(G_centered)
        # project the data points to span{eigen_vectors}
        Z = G_centered @ eigen_vectors
        g_star = np.array([self.kernel(self.X[i], self.X[0]) for i in range(ns)])
        z = eigen_vectors.T @ MyKernelPCA.__center_g(g_star, self.G)
        if any(~np.isclose(z.ravel(), Z[0, :])):  # NOTE: This should not happen!
            breakpoint()
        Z_std = np.sqrt(eigen_values / ns)
        # compute the absolute correlation between each eigenvector to the objective values
        correlation = np.abs(np.mean(Z * y_, axis=0) / Z_std)
        correlation[np.isinf(correlation)] = 0
        total_correlation = np.sum(correlation)
        # sort the eigen_vectors according to the absolute correlation
        idx = np.argsort(correlation)[::-1]
        correlation = correlation[idx]
        eigen_vectors = eigen_vectors[:, idx]
        # select components based on the explained correlation
        _cumsum = np.cumsum(correlation)
        self.k = np.nonzero(_cumsum >= (1.0 - self.epsilon) * total_correlation)[0][0] + 1
        self.extracted_information = _cumsum[self.k] / total_correlation
        self.V = eigen_vectors[:, 0 : self.k].T

    def transform(self, X: np.ndarray) -> np.ndarray:
        gamma = self.kernel_config["kernel_parameters"]["gamma"]
        n_point = len(X)
        Z = np.zeros((n_point, self.k))
        for i, x in enumerate(X):
            dist = cdist(self.X, x[np.newaxis, :]) ** 2
            g = np.exp(-gamma * dist)
            Z[i, :] = (self.V @ MyKernelPCA.__center_g(g, self.G)).ravel()
        return Z

    def get_good_subspace(self):
        p = [i for i in range(len(self.X))]
        for i in range(len(p)):
            ind = random.randint(0, i)
            p[ind], p[i] = p[i], p[ind]
        return [self.X[i] for i in p[: self.NN]]

    def inverse_transform(self, Y: np.ndarray):
        if not hasattr(self, "k"):
            return Y
        if not len(Y.shape) == 2:
            raise ValueError("Y array should be at least 2d but got this instead", Y)

        Y_inversed = []
        for y in Y:
            if not len(y) == self.k:
                raise ValueError(
                    f"dimensionality of point is supposed to be {self.k}, but it is {len(y)}, the point {y}"
                )
            good_subspace = self.get_good_subspace()
            partial_f = partial(
                MyKernelPCA.hparam_cost_func,
                X=self.X,
                good_subspace=good_subspace,
                kernel=self.kernel,
                z_star=y,
                G=self.G,
                V=self.V,
                bounds=self.bounds,
            )

            fopt = np.inf
            initial_weights = np.random.randn(50, len(good_subspace))
            v = [partial_f(w) for w in initial_weights]
            idx = np.argsort(v)
            initial_weights = initial_weights[idx, :]

            for i in range(5):
                w = initial_weights[i, :]
                w0_, fopt_, *rest = optimize.fmin_bfgs(partial_f, w, full_output=True, disp=False)
                if fopt_ < fopt:
                    w0 = w0_
                    fopt = fopt_

            inversed = MyKernelPCA.linear_combination(w0, good_subspace)
            ns = len(self.X)
            g_star = np.array([self.kernel(self.X[i], inversed) for i in range(ns)])
            z = (self.V @ MyKernelPCA.__center_g(g_star, self.G)).ravel()
            Y_inversed.append(inversed)
            eprintf(f"Inverse of point {y} is {inversed}")

        return np.array(Y_inversed)
