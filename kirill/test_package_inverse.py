from bayes_optim.extension import *
from bayes_optim.mylogging import *
import benchmark.bbobbenchmarks as bn
import random
import numpy as np

def run_experiment():
    dim = 2
    lb, ub = -5, 5
    N = 1
    space = RealSpace([lb, ub]) * dim
    myKernelPCA = KernelPCA(kernel="rbf", gamma=0.0001)
    X_doe = space._sample(5)
    # X_sampled = np.concatenate((space._sample(100), X_doe), axis=0)
    X_sampled = X_doe
    X_transformed = myKernelPCA.fit_transform(X_sampled)
    
    # Inverse Transformer Learning
    krr = KernelRidge(kernel=myKernelPCA.kernel,
                      kernel_params=myKernelPCA.get_params())
    inverser = InverseTransformKPCA(X_sampled, krr, space)
    inverser.fit(X_transformed[:, 0:N])

    # Inverse Transformer Application
    X_doe_1 = [[0. for _ in range(len(X_doe))] for _ in range(dim)]
    cnt = 0
    for x in X_doe:
        y = myKernelPCA.transform([x])[:,0:N]
        x1 = inverser.inverse(y[0])
        for i in range(dim):
            X_doe_1[i][cnt] = x1[i]
        eprintf("x:", x, "\Phi(x):", y, "\Phi^{-1}(\Phi(x)):", x1)
        cnt += 1  
    

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    run_experiment()


