from bayes_optim.extension import *
from bayes_optim.mylogging import *
import benchmark.bbobbenchmarks as bn
import random
import numpy as np
from kirill.utils import *


def run_experiment():
    dim = 2
    lb, ub = -5, 5
    N = 1
    space = RealSpace([lb, ub],random_seed=0) * dim
    myKernelPCA = KernelPCA(kernel="rbf", gamma=0.01)
    X_doe = space._sample(5)
    X_sampled = np.concatenate((space._sample(100), X_doe), axis=0)
    X_transformed = myKernelPCA.fit_transform(X_sampled)

    # Inverse Transformer Learning
    krr = KernelRidge(kernel=myKernelPCA.kernel,
            kernel_params=myKernelPCA.get_params())
    inverser = InverseTransformKPCA(X_sampled, krr, space)
    inverser.fit(X_transformed)
    
    centre, k = find_k_for_rbf(myKernelPCA, space)
    eprintf("k for space is", k)

    # Inverse Transformer Application
    eprintf("X_doe\n", X_doe)
    for x in [X_doe[1]]:
        y = myKernelPCA.transform([x])[0]
        alpha = find_alpha(y, X_transformed)
        eprintf("alpha vector", alpha)
        eprintf("initial y", y, "linear approximation", np.matmul(alpha, X_transformed))
        g = gram(myKernelPCA, centre, X_sampled)
        eprintf("gram line", g)
        innerProduct = np.dot(alpha, g)
        eprintf("inner product", innerProduct)
        yyInnerProduct = get_yy_inner(myKernelPCA, alpha, X_sampled)
        eprintf("yy inner product", yyInnerProduct)
        d = 1. + yyInnerProduct - 2.*innerProduct
        eprintf("Without information about the pre-image d_H(\Phi(centre), y) =", d, "and k is", k)
        eprintf("With kernels d_H(\Phi(centre), y = \Phi(x)) =", 2. * (1. - kernel_function(myKernelPCA, centre, x)))
        find_inverse(x, myKernelPCA.transform, inverser.inverse)
    # find_inverse([-4.99,-4.99], myKernelPCA.transform, inverser.inverse) 

    # eprintf("---------------- Points outside of the domain -------------------")
    # X_outside = [[6., 10.],[-6,-6], [-5.4, 5.]]
    # for x in X_outside:
        # find_inverse(x, myKernelPCA.transform, inverser.inverse)

def run_in_out_experiment():
    dim = 2
    space = RealSpace([-5, 5],random_seed=0) * dim
    X,values,col = sample_doe(-5,5,dim,100,bn.F17())
    X_weighted = get_rescaled_points(X, values)
    myKernelPCA = KernelPCA(kernel="rbf", gamma=0.001)
    myKernelPCA.fit(X_weighted)
    Y = myKernelPCA.transform(X)
    krr = KernelRidge(kernel=myKernelPCA.kernel,
            kernel_params=myKernelPCA.get_params())
    inverser = InverseTransformKPCA(X, krr, space)
    inverser.fit(Y)
    eprintf("---Points in bounding box---")
    ma = -1
    space1 = RealSpace([-3, 3],random_seed=0) * dim
    for x in space1._sample(100):
        y = myKernelPCA.transform([x])[0]
        inverser.create_f(y)
        value = inverser.eval_function([0,0])
        eprintf("Value", value)
        eprintf("Value in x", inverser.eval_function(x))
        ma = max(ma, value)
    eprintf("Max value in", ma)
    eprintf("---Points out of bounding box")
    space1 = RealSpace([5, 10],random_seed=0)*dim
    space2 = RealSpace([-10, -5], random_seed=0)*dim
    X_sampled = np.concatenate((space1._sample(50), space2._sample(50)), axis=0)
    mi = 1e9
    for x in X_sampled:
        y = myKernelPCA.transform([x])[0]
        inverser.create_f(y)
        value = inverser.eval_function([0,0])
        eprintf("Value", value)
        mi = min(mi, value)
    eprintf("Min value out", mi)


def find_k_for_rbf(model, space):
    centre = [0.] * space.dim
    corner = [0.] * space.dim
    for i in range(len(space.bounds)):
        centre[i] = (min(space.bounds[i]) + max(space.bounds[i])) / 2
        corner[i] = max(space.bounds[i])
    eprintf("centre is", centre, "corner is", corner)
    return centre, 2 * (1. - kernel_function(model, centre, corner))

def get_yy_inner(model, alpha, X_sampled):
    s = 0.
    for i in range(len(X_sampled)):
        for j in range(len(X_sampled)):
            s += alpha[i] *alpha[j] * kernel_function(model, X_sampled[i], X_sampled[j])
    return s

def gram(model, centre, X):
    g = np.zeros(len(X))
    for i in range(len(X)):
        g[i] = kernel_function(model, centre, X[i])
    return g


def find_alpha(y, phi):
    return np.matmul(y, np.linalg.pinv(phi))

def find_inverse(x, T, Ti):
    y = T([x])
    x1 = Ti(y[0])
    eprintf("x:", x, "\Phi(x):", y, "\Phi^{-1}(\Phi(x)):", x1)


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    run_in_out_experiment()


