import matplotlib.pyplot as plt
from bayes_optim import RealSpace
from bayes_optim.mylogging import *
from bayes_optim.kpca import *
from kirill.utils import *


SAVER = PictureSaver('./', '17', 'png')


def run_experiment():
    dim = 2
    lb, ub = -5, 5
    DOESIZE = 20

    X, Y, colours = sample_doe(lb, ub, dim, DOESIZE, bn.F17())
    X_weighted = get_rescaled_points(X, Y)
    kpca = MyKernelPCA([], 'rbf', 0.1, {'gamma': 0.001})
    kpca.fit(X_weighted)
    kpca.set_initial_space_points(X)

    spaceTest = RealSpace([lb*2, ub*2], random_seed=0)*dim
    for point_number in range(10):
        x = spaceTest._sample(1)[0]
        y = kpca.transform([x])[0]
        x1 = kpca.inverse_transform(np.array([y]))[0]
        eprintf(f'Initial {x}, Feature Space {y}, Restored {x1}')


def restore_initial_space_experiment():
    dim = 2
    lb, ub = -5, 5
    DOESIZE = 10

    X, Y, colours = sample_doe(lb, ub, dim, DOESIZE, bn.F17())
    XT = list(map(list, zip(*X)))
    fdoe = plt.figure()
    plt.title("Original space")
    plt.xlim([lb, ub])
    plt.ylim([lb, ub])
    plt.scatter(XT[0], XT[1], c=colours)
    SAVER.save(fdoe, 'original')
    X_weighted = get_rescaled_points(X, Y)
    kpca = MyKernelPCA([], 'rbf', 0.4, kernel_params_dict={'gamma': 1, 'd': 2, 'c0': 0})
    kpca.set_initial_space_points(X)
    kpca.fit(X_weighted)

    spaceTest = RealSpace([lb*2, ub*2], random_seed=0)*dim
    X1 = []
    Y = []
    for x in X:
        y = kpca.transform([x])[0]
        # eprintf(f'Initial point is {x}, Feature space point is {y}')
        x1 = kpca.inverse_transform(np.array([y]))[0]
        X1.append(x1)
        eprintf(f'Initial {x}, Feature Space {y}, Restored {x1}, Feature Space of the restored {kpca.transform([x1])[0]}')
    # X1T = list(map(list, zip(*X1)))
    X1T = get_transpose(X1)
    frestored = plt.figure()
    plt.title('Restored')
    plt.xlim([lb, ub])
    plt.ylim([lb, ub])
    plt.scatter(X1T[0], X1T[1], c=colours)
    SAVER.save(frestored, 'restored')

    eprintf("Weighted points", Y)
    fweighted = plt.figure()
    plt.title('Weighted')
    # plt.xlim([lb, ub])
    # plt.ylim([lb, ub])
    X1_weighted = get_transpose(X_weighted)
    plt.scatter(X1_weighted[0], X1_weighted[1], c=colours)
    SAVER.save(fweighted, 'weighted')

    fall = plt.figure()
    plt.title('Black - initial, Red - restored')
    # plt.xlim([lb, ub])
    # plt.ylim([lb, ub])
    plt.scatter(XT[0], XT[1], c='black')
    plt.scatter(X1T[0], X1T[1], c='red')
    SAVER.save(fall, 'all')


def gogogo(x1):
    dim = 2
    lb, ub = -5, 5
    DOESIZE = 10

    X, Y, colours = sample_doe(lb, ub, dim, DOESIZE, bn.F17())
    XT = list(map(list, zip(*X)))
    fdoe = plt.figure()
    plt.title("Original space")
    plt.xlim([lb, ub])
    plt.ylim([lb, ub])
    plt.scatter(XT[0], XT[1], c=colours)
    SAVER.save(fdoe, 'original')
    X_weighted = get_rescaled_points(X, Y)
    kpca = MyKernelPCA([], 'rbf', 0.4, {'gamma': 0.001})
    kpca.set_initial_space_points(X)
    kpca.fit(X)

    y = [0.03685932]
    eprintf(f"Target image is {y}")
    x_true_preimage = [3.4442185152504816, 2.5795440294030243]
    eprintf(f"True pre-image of {y} is {x_true_preimage}")
    found = kpca.inverse_transform(np.array([y]))[0]
    eprintf(f"Found pre-image of {y} is {found}, the image is {kpca.transform([found])[0]}")
    # eprintf(f"Image of the true pre-image is {kpca.transform([x_true_preimage])[0]}")
    # x1 = [3.1, 2.1]
    eprintf(f"Target Image is {y}, obtained image of point {x1} is {kpca.transform([x1])[0]}")


def weighting_scheme(X, y):
    self.center = X.mean(axis=0)
    X_centered = X - self.center
    y_ = -1 * y if not self.minimize else y
    r = rankdata(y_)
    N = len(y_)
    w = np.log(N) - np.log(r)
    w /= np.sum(w)
    X_scaled = X_centered * w.reshape(-1, 1)
    # eprintf("Before scaling", X_scaled)
    # d_initial = self._find_max_dist(X)
    # d_weighted = self._find_max_dist(X_scaled)
    # scaling_factor = np.sqrt(d_initial / d_weighted)
    # for i in range(len(X_scaled)):
        # for j in range(len(X_scaled[0])):
            # X_scaled[i][j] *= scaling_factor
    # eprintf("After scaling", X_scaled)
    return X_scaled


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    passed_point = [float(number) for number in sys.argv[1:]]
    restore_initial_space_experiment()

