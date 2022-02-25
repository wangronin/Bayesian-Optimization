from .mylogging import *
import numpy as np
from functools import partial 
from scipy import optimize
from math import exp
from copy import deepcopy


KERNEL_PARAMETERS = {}

def additive_chi2_kernel(a, b):
    # TODO
    raise NotImplementedError

def chi2_kernel(a,b):
    # TODO
    raise NotImplementedError

def linear_kernel(a,b):
    # TODO
    raise NotImplementedError

def polynomial_kernel(a,b):
    # TODO
    raise NotImplementedError

def rbf_kernel(a, b):
    gamma = KERNEL_PARAMETERS['gamma']
    return exp(-gamma*np.sum((np.array(a)-np.array(b))**2))

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

class MyKernelPCA:
    # Implementation is based on paper García_González_et_al_2021_A_kernel_Principal_Component_Analysis
    def __init__(self, X_initial_space, kernel_name, epsilon, kernel_params_dict):
        if kernel_name not in PAIRWISE_KERNEL_FUNCTIONS:
            raise ValueError(f'There is no kernel with name {kernel_name}')
        global KERNEL_PARAMETERS
        for k,v in kernel_params_dict.items():
            KERNEL_PARAMETERS[k] = v
        self.kernel_name = kernel_name
        self.kernel = PAIRWISE_KERNEL_FUNCTIONS[kernel_name]
        self.epsilon = epsilon
        self.X_initial_space = X_initial_space
        self.NN = 5
        
    def set_initial_space_points(self, X):
        self.X_initial_space = X

    def __center_G(self, G):
        ns = len(G)
        line = [0.] * len(G)
        for i in range(len(G)):
            line[i] = sum(G[i])
        all_sum = sum(line)
        return [[G[i][j] - line[i]/ns - line[j]/ns + all_sum/ns**2 for j in range(len(G))] for i in range(len(G))]

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
    def f(X,k,V,z_star,w):
        candidate_x = MyKernelPCA.linear_combination(w, X)
        g_star = [0.] * len(X)
        for i in range(len(X)):
            g_star[i] = k(X[i], candidate_x)
        return MyKernelPCA.l2(np.transpose(np.array(z_star)) - np.matmul(V, np.array(g_star))) 

    @staticmethod
    def linear_combination(w, X):
        comb = [0.] * len(X[0])
        for i in range(len(X)):
            for j in range(len(X[0])):
                comb[j]+=w[i]*X[i][j]
        return comb

    def fit(self, X_weighted: np.ndarray):
        G = [[self.kernel(x1, x2) for x1 in X_weighted] for x2 in X_weighted]
        G_centered = self.__center_G(G)
        eignValues, eignVectors = self.__sorted_eig(G_centered)
        eignValuesSum = sum(eignValues)
        s = 0
        self.k = 0
        while s<(1.-self.epsilon)*eignValuesSum:
            eprintf(f"percentage of first {self.k} components is {s/eignValuesSum*100.}")
            s += eignValues[self.k]
            self.k += 1
        eprintf(f"percentage of first {self.k} components is {s/eignValuesSum*100.}")
        eprintf("k equals to", self.k)
        V = np.transpose(eignVectors)
        self.V = V[:self.k]
    
    def transform(self, X: np.ndarray):
        X_gram_lines = []
        for x in X:
            X_gram_lines.append(self.__get_gram_line(self.X_initial_space, x))
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
        eprintf("V\n", self.V)
        V1 = deepcopy(self.V[:,:sz])
        for i in range(sz):
            V1[:,i] = self.V[:,dists[i][1]]
        eprintf("V1\n", V1)
        return good_subspace, V1


    def inverse_transform(self, Y: np.ndarray):
        if not hasattr(self, "k"):
            return Y
        eprintf("attribute found")
        if not len(Y.shape) == 2:
            raise ValueError("Y array should be at least 2d but got this instead", Y)
        Y_inversed = []
        for y in Y:
            if not len(y) == self.k:
                raise ValueError(f"dimensionality of point is supposed to be {self.k}, but it is {len(y)}")
            eprintf("point to find inverse", y)
            # good_subspace, V1 = self.get_good_subspace(y)
            good_subspace, V1 = self.X_initial_space, self.V
            partial_f = partial(MyKernelPCA.f, good_subspace, self.kernel, V1, y)
            w0, fopt, *rest = optimize.fmin_bfgs(partial_f, np.zeros(len(good_subspace)), full_output=True, disp=False)
            inversed = MyKernelPCA.linear_combination(w0, good_subspace)
            Y_inversed.append(inversed)
            eprintf(f"the final value of J is {fopt}")
            eprintf(f"the inverse of point {y} is {inversed}")
        return np.array(Y_inversed)

