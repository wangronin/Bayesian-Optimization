from bayes_optim.extension import *
from bayes_optim.mylogging import *
import benchmark.bbobbenchmarks as bn
import random
import numpy as np
from kirill.utils import *
from functools import partial 
from scipy import optimize


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


def run_experiment():
    dim = 2
    lb, ub = -5, 5
    DOESIZE = 5
    space = RealSpace([lb, ub],random_seed=0) * dim
    kpca = KernelPCA(kernel="rbf", gamma=0.01)
    X, Y, colours = sample_doe(lb, ub, dim, DOESIZE, bn.F17())
    # X_weighted = get_rescaled_points(X, Y)
    X_weighted = X
    kpca.fit(X_weighted)
    
    G = [[kernel_function(kpca, x1, x2) for x1 in X_weighted] for x2 in X_weighted]
    G1 = center_G(G)
    eprintf("G is\n", np.array(G))
    eprintf("G1 is\n", np.array(G1))

    eigenValues, eignVectors = np.linalg.eig(G1)

    V = np.transpose(eignVectors)
    k = 4
    eprintf("V", V)
    V = V[:k]
    eprintf("\n", V)
    
    TEST = 2
    for i in range(TEST):
        if i==1:
            p=[x1]
        else:
            p = space._sample(1)
        eprintf(p)
        y = kpca.transform(p)[0]
        partial_f = partial(f, X, partial(kernel_function, kpca), V, y, k)
        w0, fopt,*rest = optimize.fmin_bfgs(partial_f, np.zeros(len(X)), full_output=True, disp=False)
        eprintf("fopt", fopt)
        eprintf("w0", w0)
        eprintf(linear_combination(w0, X))
        x1 = linear_combination(w0, X)


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

