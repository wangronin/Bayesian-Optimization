from .mylogging import *
import numpy as np
from functools import partial
from scipy import optimize
from math import exp
from copy import deepcopy


def additive_chi2_kernel(a, b):
    # TODO
    raise NotImplementedError

def chi2_kernel(a,b):
    # TODO
    raise NotImplementedError

def linear_kernel(a,b):
    # TODO
    raise NotImplementedError

def polynomial_kernel(a, b, gamma, d, c0, **kwargs):
    return (sum(ai * bi * gamma for ai, bi in zip(a, b)) + c0) ** d

def rbf_kernel(a, b, gamma, **kwargs):
    return exp(-gamma * np.sum((np.array(a) - np.array(b)) ** 2))

def laplacian_kernel(a,b):
    # TODO
    raise NotImplementedError

def sigmoid_kernel(a,b):
    # TODO
    raise NotImplementedError

def cosine_similarity(a,b):
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
    if kernel_name == '__internal_rbf':
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

    def set_initial_space_points(self, X):
        self.X_initial_space = X

    def __center_G(self, G):
        ns = len(G)
        line = [0.] * len(G)
        for i in range(len(G)):
            line[i] = sum(G[i])
        all_sum = sum(line)
        return [[G[i][j] - line[i]/ns - line[j]/ns + all_sum/ns**2 for j in range(len(G[i]))] for i in range(len(G))]

    @staticmethod
    def __center_gram_line(g):
        delta = sum(g) / len(g)
        for i in range(len(g)):
            g[i] -= delta
        return g

    def __sorted_eig(self, X):
        values, vectors = np.linalg.eig(X)
        values_ids = [(v,i) for i,v in enumerate(values)]
        values_ids.sort()
        values_ids = values_ids[::-1]
        sorted_vectors = deepcopy(vectors)
        sorted_values = deepcopy(values)
        cnt = 0
        for v, i in values_ids:
            for j in range(len(vectors)):
                sorted_vectors[j][cnt] = vectors[j][i]
            cnt += 1
        for i in range(len(values)):
            sorted_values[i],_ = values_ids[i]
        return sorted_values, sorted_vectors

    def __get_gram_line(self, X, p):
        return np.array([self.kernel(p, x) for x in X])

    @staticmethod
    def l2(x):
        ans = 0
        for i in range(len(x)):
            ans += x[i]**2
        return ans

    @staticmethod
    def f(X,good_subspace,k,V,z_star,w):
        candidate_x = MyKernelPCA.linear_combination(w, good_subspace)
        g_star = [0.] * len(X)
        for i in range(len(X)):
            g_star[i] = k(X[i], candidate_x)
        g_star = MyKernelPCA.__center_gram_line(g_star)
        return sum((np.transpose(np.array(z_star)) - np.matmul(V, np.array(g_star)))**2)

    @staticmethod
    def linear_combination(w, X):
        comb = [0.] * len(X[0])
        for i in range(len(X)):
            for j in range(len(X[0])):
                comb[j]+=w[i]*X[i][j]
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
        return too_compressed_points_cnt > int(len(G_centered)/2)

    def _set_reuse_rbf_data(self, l2_norm_matrix):
        kernel_name = self.kernel_config['kernel_name']
        if  kernel_name != 'rbf':
            raise ValueError(f'rbf kernel is expected, but {kernel_name} found')
        self._reuse_rbf_data = True
        self._l2_norm_matrix = l2_norm_matrix

    def __compute_G(self, X_weighted):
        if self._reuse_rbf_data:
            G = self._l2_norm_matrix
            gamma = self.kernel_config['kernel_parameters']['gamma']
            return [[exp(-gamma * gij) for gij in Gi] for Gi in G]
        return [[self.kernel(x1, x2) for x1 in X_weighted] for x2 in X_weighted]

    def fit(self, X_weighted: np.ndarray):
        self.NN = len(X_weighted[0]) + 1
        self.kernel = create_kernel(self.kernel_config['kernel_name'], self.kernel_config['kernel_parameters'])
        self.X_weighted = X_weighted
        G = self.__compute_G(X_weighted)
        G_centered = self.__center_G(G)
        self.too_compressed = self.__is_too_compressed(G_centered)
        eignValues, eignVectors = self.__sorted_eig(G_centered)
        eignValues[eignValues < 0] = 0
        for e in eignValues:
            if e < 0:
                breakpoint()
        eignValues = eignValues.view(np.float64)
        eignVectors = eignVectors.view(np.float64)
        eignValuesSum = sum(t for t in eignValues)
        s = 0
        self.k = 0
        while s<(1.-self.epsilon)*eignValuesSum:
            s += eignValues[self.k]
            self.k += 1
        self.extracted_information = s / eignValuesSum
        V = np.transpose(eignVectors)
        self.V = V[:self.k]

    def transform(self, X: np.ndarray):
        X_gram_lines = []
        for x in X:
            g = self.__get_gram_line(self.X_initial_space, x)
            g = self.__center_gram_line(g)
            X_gram_lines.append(g)
        M = np.transpose(X_gram_lines)
        return np.transpose((self.V @ M)[:self.k])

    def get_good_subspace(self, y):
        Y = self.transform(self.X_initial_space)
        dists = [[0., i] for i in range(len(Y))]
        for i in range(len(Y)):
            dists[i][0] = MyKernelPCA.l2(y - Y[i])
        dists.sort()
        sz = min(self.NN, len(self.X_initial_space))
        good_subspace = [[] for i in range(sz)]
        for i in range(sz):
            good_subspace[i] = self.X_initial_space[dists[i][1]]
        V1 = deepcopy(self.V[:,:sz])
        for i in range(sz):
            V1[:,i] = self.V[:,dists[i][1]]
        return good_subspace, V1

    def inverse_transform(self, Y: np.ndarray):
        if not hasattr(self, "k"):
            return Y
        if not len(Y.shape) == 2:
            raise ValueError("Y array should be at least 2d but got this instead", Y)
        Y_inversed = []
        for y in Y:
            if not len(y) == self.k:
                raise ValueError(f"dimensionality of point is supposed to be {self.k}, but it is {len(y)}, the point {y}")
            good_subspace, V1 = self.get_good_subspace(y)
            # good_subspace, V1 = self.X_initial_space, self.V
            partial_f = partial(MyKernelPCA.f, self.X_initial_space, good_subspace, self.kernel, self.V, y)
            initial_weights = np.zeros(len(good_subspace))
            w0, fopt, *rest = optimize.fmin_bfgs(partial_f, initial_weights, full_output=True, disp=False)
            inversed = MyKernelPCA.linear_combination(w0, good_subspace)
            Y_inversed.append(inversed)
            eprintf(f"Inverse of point {y} is {inversed}")

        return np.array(Y_inversed)

