from kirill.utils import *
import sys
import os
import functools
import math
from scipy import optimize
from bayes_optim.acquisition_optim import OnePlusOne_Cholesky_CMA
from bayes_optim.extension import RealSpace
from bayes_optim.mylogging import eprintf


SEED = 0


def run():
    obj_f = bn.F17()
    KPCA = KernelPCA(kernel="rbf", gamma=float(sys.argv[1]))
    X, values, colours = sample_doe(-5, 5, 2, 20, obj_f)
    X_weighted = get_rescaled_points(X, values)
    KPCA.fit(X_weighted)
    Y = KPCA.transform(X)
    directory = './F' + str(obj_f.funId) + 'K' + KPCA.kernel + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    saver = PictureSaver(directory, str(obj_f.funId), 'png')

    fig = plt.figure()
    plt.title(f'Weighted points {KPCA.kernel}, $gamma$: {KPCA.gamma}')
    Xw = get_transpose(X_weighted)
    plt.scatter(Xw[0], Xw[1], c=colours)
    saver.save(fig, 'weighted')

    fig = plt.figure()
    plt.title(f'Feature space, kernel: {KPCA.kernel}, $gamma$: {KPCA.gamma}')
    plt.scatter(Y[:, 0], Y[:, 1], c=colours)
    saver.save(fig, 'gamma')
    
    fig = plt.figure()
    var = get_sorted_var_columns_pairs(Y)
    print(sum(p for (p,_) in var))
    plt.bar(np.arange(len(var)), [a for (a,b) in var])
    plt.ylabel("$ {\sigma^2_i}/{\sum \sigma^2_i}$")
    plt.xlabel("$\sigma^2_i$")
    plt.title(f"Sorted variances bar chart, kernel: {KPCA.kernel}, $gamma$: {KPCA.gamma}")
    saver.save(fig, 'variance')


def run1():
    obj_f = bn.F21()
    directory = './F' + str(obj_f.funId) + 'K' + 'rbf' + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    saver = PictureSaver(directory, str(obj_f.funId), 'png')
    X, values, colours = sample_doe(-5, 5, 2, 50, obj_f)
    X_weighted = get_rescaled_points(X, values)

    gamma = get_kernel_parameters(1, X, X_weighted)
    KPCA = KernelPCA(kernel="rbf", gamma=gamma)

    KPCA.fit(X_weighted)
    Y = KPCA.transform(X)

    fig = plt.figure()
    var = get_sorted_var_columns_pairs(Y)
    plt.title(f'Feature space, kernel: {KPCA.kernel}, $gamma$: {KPCA.gamma}')
    _, pc1_index = var[0]
    _, pc2_index = var[1]
    plt.scatter(Y[:, -pc1_index], Y[:, -pc2_index], c=colours)
    saver.save(fig, 'gamma-best')
    
    fig = plt.figure()
    print('variances', var[0:min(10,len(var))])
    print(sum(p for (p,_) in var))
    plt.bar(np.arange(len(var)), [a for (a,b) in var])
    plt.ylabel("$ {\sigma^2_i}/{\sum \sigma^2_i}$")
    plt.xlabel("$\sigma^2_i$")
    plt.title(f"Sorted variances bar chart, kernel: {KPCA.kernel}, $gamma$: {KPCA.gamma}")
    saver.save(fig, 'variance-best')


def test_gamma(gamma, n_components, X, X_weighted):
    KPCA = KernelPCA(kernel="rbf", gamma=gamma[0])
    KPCA.fit(X_weighted)
    Y = KPCA.transform(X)
    Y = np.array(Y)
    variances = [0.] * len(Y[0])
    for i in range(len(Y[0])):
        variances[i] = statistics.variance(Y[:, i])
    variances.sort()
    variances.reverse()
    value = sum(v for v in variances[:n_components]) / sum(v for v in variances)
    print('gamma', gamma, 'value', value)
    return -value


def exponential_grid_minimizer(f, start, end, steps):
    t1 = math.log(start)
    t2 = math.log(end)
    eps = (t2 - t1)/100
    mi, argmi = 1., 0.
    for i in range(steps):
        gamma = math.pow(math.e, t1 + i*eps)
        value = f([gamma])
        if value < mi:
            mi = value
            argmi = gamma
    return argmi


def get_kernel_parameters(n_components, X, X_weighted):
    f = functools.partial(
        test_gamma,
        n_components=n_components,
        X=X,
        X_weighted=X_weighted
    )
#    opt = OnePlusOne_Cholesky_CMA(dim=1, obj_fun=f, search_space=RealSpace([1e-3, 100], random_seed=SEED), x0=[0.5], ftarget=-1., n_restarts=5, minimize=True, verbose=True)
#    x,_,_=opt.run()
#    minimized = optimize.minimize(f, x0=[0.5], method='CG')
    x = exponential_grid_minimizer(f, 0.00001, 10., 100)
    return x



if __name__ == '__main__':
    random.seed(SEED)
    run1()

