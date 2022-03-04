from bayes_optim import RealSpace
from bayes_optim.mylogging import *
import benchmark.bbobbenchmarks as bn
import random
import numpy as np
from kirill.utils import *
from functools import partial 
from scipy import optimize
from math import exp


def l2(x):
    ans = 0
    for i in range(len(x)):
        ans += x[i]**2
    return ans

def linear_combination(w, X):
    comb = [0.] * len(X[0])
    for i in range(len(X)):
        for j in range(len(X[0])):
            comb[j]+=w[i]*X[i][j]
    return comb


def f(X,k,V,z_star,sz,w):
    candidate_x = linear_combination(w, X)
    g_star = [0.] * len(X)
    for i in range(len(X)):
        g_star[i] = k(X[i], candidate_x)
    return l2(np.transpose(np.array(z_star[:sz])) - np.matmul(V, np.array(g_star))) 


def sorted_eig(X):
    values, vectors = np.linalg.eig(X)
    values_ids = [(v,i) for i,v in enumerate(values)]
    values_ids.sort()
    values_ids = values_ids[::-1]
    sorted_vectors = deepcopy(vectors)
    sorted_values = deepcopy(values)
    cnt = 0
    for v,i in values_ids:
        for j in range(len(vectors)):
            sorted_vectors[j][cnt] = vectors[j][i]
        cnt += 1
    for i in range(len(values)):
        sorted_values[i],_ = values_ids[i]
    return sorted_values, sorted_vectors


def rbf(gamma, a, b):
    return exp(-gamma*np.sum((np.array(a)-np.array(b))**2))


def get_gram_line(k, X, p):
    return np.array([k(p, x) for x in X])


def run_experiment():
    dim = 2
    lb, ub = -10, 10
    DOESIZE = 5
    space = RealSpace([lb, ub],random_seed=0) * dim
    # kpca = KernelPCA(kernel="rbf", gamma=0.01)
    X, Y, colours = sample_doe(lb, ub, dim, DOESIZE, bn.F17())
    print("DoE\n", X)
    print("Y\n", Y)
    X_weighted = get_rescaled_points(X, Y)
    eprintf("X_weighted\n", X_weighted)
    # X_weighted = X
    # kpca.fit(X_weighted)
    my_kernel = partial(rbf, 0.01)
    
    G = [[my_kernel(x1, x2) for x1 in X_weighted] for x2 in X_weighted]
    G1 = center_G(G)
    eprintf("G is\n", np.array(G))
    eprintf("G1 is\n", np.array(G1))

    eignValues, eignVectors = sorted_eig(G1)
    eprintf("eigen values", eignValues)
    eprintf("sum of eigenvalues is", sum(eignValues))
    epsilon = 0.2
    eignValuesSum = sum(eignValues)
    s=0
    k=0
    while s<=(1-epsilon)*eignValuesSum:
        eprintf(f"percentage of first {k} components is {s/eignValuesSum*100.}")
        s += eignValues[k]
        k += 1
    
    eprintf(f"percentage of first {k} components is {s/eignValuesSum*100.}")
    eprintf("k equals to", k)
    eignValueMatrix = deepcopy(eignVectors)
    for i in range(len(eignValueMatrix)):
        for j in range(len(eignValueMatrix)):
            if i != j: eignValueMatrix[i][j] = 0.
            else: eignValueMatrix[i][j] = eignValues[i]

    eprintf("V*L*V^T should be equal to G1:\n", np.matmul(np.matmul(eignVectors, eignValueMatrix), np.transpose(eignVectors)))

    V = np.transpose(eignVectors)
    eprintf("V\n", V)
    V = V[:k]
    eprintf("V reduced\n", V)
    
    TEST = 2
    for i in range(TEST):
        if i == 0:
            p = space._sample(1)
        else:
            p = x1
        # eprintf('initial point is', p)
        # y = kpca.transform(p)[0]
        g = get_gram_line(my_kernel, X, p)
        # eprintf("g(p) is ", g)
        y = (V @ np.transpose(g))[:k]
        # eprintf("point in the feature space", y)
        partial_f = partial(f, X, my_kernel, V, y, k)
        w0, fopt,*rest = optimize.fmin_bfgs(partial_f, np.zeros(len(X)), full_output=True, disp=False)
        # eprintf("fopt", fopt)
        # eprintf("w0", w0)
        p1 = linear_combination(w0, X)
        eprintf('initial point is', p, 'I(\Phi(x)) is', p1)
        x1 = p1


def center_G(G):
    ns = len(G)
    line = [0.] * len(G)
    for i in range(len(G)):
        line[i] = sum(G[i])
    all_sum = sum(line)
    return [[G[i][j] - line[i]/ns - line[j]/ns + all_sum/ns**2 for j in range(len(G))] for i in range(len(G))]


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    run_experiment()

