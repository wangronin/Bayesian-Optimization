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
    def __init__(self, epsilon, X_initial_space, kernel_config, dimensions: int = None, NN: int = None):
        self.kernel_config = kernel_config
        self.epsilon = epsilon
        self.X_initial_space = X_initial_space
        self.NN = dimensions if NN is None else NN
        self._reuse_rbf_data = False

    def enable_inverse_transform(self, bounds):
        self.bounds = bounds

    def set_initial_space_points(self, X):
        self.X_initial_space = np.array(X.tolist())

    @staticmethod
    def __center_gram_line(g):
        delta = sum(g) / len(g)
        for i in range(len(g)):
            g[i] -= delta
        return g

    @staticmethod
    def l2(x):
        ans = 0
        for i in range(len(x)):
            ans += x[i] ** 2
        return ans

    @staticmethod
    def f(conical_w, X, good_subspace, k, V, z_star, bounds, w, M):
        ns = len(X)
        x_ = MyKernelPCA.linear_combination(conical_w, good_subspace)
        g_star = np.array([k(X[i], x_) for i in range(ns)])
        z = MyKernelPCA.get_z_from_g(g_star, w, ns, V, M)
        bounds_ = np.atleast_2d(bounds)
        idx_lower = np.where(x_ < bounds_[:, 0])[0]
        idx_upper = np.where(x_ > bounds_[:, 1])[0]
        penalty = np.sum([bounds_[i, 0] - x_[i] for i in idx_lower]) + np.sum(
            [x_[i] - bounds_[i, 1] for i in idx_upper]
        )
        return np.log10(np.sum((z_star.ravel() - z) ** 2)) + penalty

    @staticmethod
    def linear_combination(w, X):
        comb = [0.0] * len(X[0])
        for i in range(len(X)):
            for j in range(len(X[0])):
                comb[j] += w[i] * X[i][j]
        return comb

    def __is_too_compressed(self, G_centered):
        EPS = 1e-4
        too_compressed_points_cnt = 0
        for i in range(len(G_centered)):
            ok = False
            for j in range(len(G_centered[i])):
                if abs(G_centered[i][j]) > EPS:
                    ok = True
                    break
            if not ok:
                too_compressed_points_cnt += 1
        res = too_compressed_points_cnt > int(len(G_centered) / 2)
        return res

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
        values[values < 0] = 0  # for numerical error
        idx = np.argsort(values)[::-1]
        return values[idx], vectors[:, idx]

    def __compute_G(self, X_weighted) -> np.ndarray:
        gamma = self.kernel_config["kernel_parameters"]["gamma"]
        D = self._l2_norm_matrix if self._reuse_rbf_data else cdist(X_weighted, X_weighted) ** 2
        return np.exp(-gamma * D)

    def __center_G(self, G: np.ndarray, w: np.ndarray) -> np.ndarray:
        ns = len(G)
        W = np.outer(w, w)
        O = np.ones((ns, ns))
        M = G @ O
        G_tilde = W * (G - M / ns - M.T / ns + M.T @ O / ns**2)
        M_prime = G_tilde @ O
        G_prime = G_tilde - M_prime / ns - M_prime.T / ns + M_prime.T @ O / ns**2
        return G_prime

    def fit(self, X_weighted: np.ndarray):
        self.kernel = create_kernel(
            self.kernel_config["kernel_name"], self.kernel_config["kernel_parameters"]
        )
        self.X, self.w = X_weighted
        # compute the raw Gram matrix
        G = self.__compute_G(self.X)
        # center and weight the Gram matrix
        G_centered = self.__center_G(G, self.w)
        # self.too_compressed = self.__is_too_compressed(G_centered)
        self.too_compressed = False
        # eigendecomposition
        eigen_values, eigen_vectors = self.__sorted_eig(G_centered)
        # select components
        total_variance = np.sum(eigen_values)
        _cumsum = np.cumsum(eigen_values)
        self.k = np.nonzero(_cumsum >= (1.0 - self.epsilon) * total_variance)[0][0] + 1
        self.extracted_information = _cumsum[self.k] / total_variance
        self.V = eigen_vectors[:, 0 : self.k].T

        # compute the variables for `transform`
        ns = len(self.X)
        I = np.ones((ns, ns))
        P = np.diag(self.w) @ (G - I @ G / ns)
        Q = P - I @ P / ns
        T = self.V @ Q
        self.M = -np.sum(T, axis=1) / ns + np.sum(T, axis=1) / ns**2 - np.sum(T * self.w, axis=1) / ns

    def get_z_from_g(g, w, ns, V, M):
        g = w * (g - np.sum(g) / ns)
        g = g - np.sum(g) / ns
        return g @ V.T + M

    def transform(self, X: np.ndarray) -> np.ndarray:
        gamma = self.kernel_config["kernel_parameters"]["gamma"]
        n_point = len(X)
        ns = len(self.X_initial_space)
        Z = np.zeros((n_point, self.k))
        for i, x in enumerate(X):
            dist = cdist(x[np.newaxis, :], self.X_initial_space) ** 2
            g = np.exp(-gamma * dist)
            Z[i, :] = MyKernelPCA.get_z_from_g(g, self.w, ns, self.V, self.M)
        return Z

    def get_good_subspace(self, y):
        p = [i for i in range(len(self.X_initial_space))]
        for i in range(len(p)):
            ind = random.randint(0, i)
            p[ind], p[i] = p[i], p[ind]
        return [self.X_initial_space[i] for i in p[: self.NN]]

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
            good_subspace = self.get_good_subspace(y)
            # good_subspace, V1 = self.X_initial_space, self.V
            partial_f = partial(
                MyKernelPCA.f,
                X=self.X_initial_space,
                good_subspace=good_subspace,
                k=self.kernel,
                V=self.V,
                z_star=y,
                bounds=self.bounds,
                w=self.w,
                M=self.M,
            )
            fopt = np.inf
            initial_weights = np.random.randn(50, len(good_subspace))
            v = [partial_f(w) for w in initial_weights]
            idx = np.argsort(v)
            initial_weights = initial_weights[idx, :]
            for i in range(5):
                # initial_weights = np.zeros(len(good_subspace))
                w = initial_weights[i, :]
                w0_, fopt_, *rest = optimize.fmin_bfgs(partial_f, w, full_output=True, disp=False)
                if fopt_ < fopt:
                    w0 = w0_
                    fopt = fopt_

            inversed = MyKernelPCA.linear_combination(w0, good_subspace)
            ns = len(self.X_initial_space)
            g_star = np.array([self.kernel(self.X_initial_space[i], inversed) for i in range(ns)])
            z = MyKernelPCA.get_z_from_g(g_star, self.w, ns, self.V, self.M)
            Y_inversed.append(inversed)
            eprintf(f"Inverse of point {y} is {inversed}")

        return np.array(Y_inversed)
