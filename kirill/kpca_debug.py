import matplotlib.pyplot as plt
from bayes_optim import RealSpace
from bayes_optim.mylogging import *
from bayes_optim.kpca import *
from kirill.utils import *

SAVER = PictureSaver('./', '17', 'png')
MY_SEED = 0

def reproduce_pattern():
    dim = 2
    lb, ub = -5, 5
    DOESIZE = 20

    X, Y, colours = sample_doe(lb, ub, dim, DOESIZE, bn.F17())
    eprintf("Initial space\n", np.array(X))
    # XT = list(map(list, zip(*X)))
    XT = get_transpose(X)
    fdoe = plt.figure()
    plt.title("Original space")
    plt.xlim([lb, ub])
    plt.ylim([lb, ub])
    plt.scatter(XT[0], XT[1], c=colours)
    SAVER.save(fdoe, 'original')
    X_weighted = get_rescaled_points(X, Y)
    kpca = MyKernelPCA([], 'rbf', 0.01, {'gamma': 0.001})
    kpca.set_initial_space_points(X)
    kpca.fit(X_weighted)

    spaceTest = RealSpace([lb*2, ub*2], random_seed=0)*dim
    X1 = []
    for i in range(DOESIZE):
        x = X[i]
        y = kpca.transform([x])[0]
        x1 = kpca.inverse_transform(np.array([y]))[0]
        X1.append(x1)
        # eprintf(f'Initial {x}, Feature Space {y}, Restored {x1}')
    # X1T = list(map(list, zip(*X1)))
    X1T = get_transpose(X1)
    frestored = plt.figure()
    plt.title('Restored')
    plt.xlim([lb, ub])
    plt.ylim([lb, ub])
    plt.scatter(X1T[0], X1T[1], c=colours)
    SAVER.save(frestored, 'restored')
    eprintf("All initial points\n", np.array(X))
    eprintf("All restored points\n", np.array(X1))

    fall = plt.figure()
    plt.title('Black - initial, Red - restored')
    # plt.xlim([lb, ub])
    # plt.ylim([lb, ub])
    plt.scatter(XT[0], XT[1], c='black')
    plt.scatter(X1T[0], X1T[1], c='red')
    SAVER.save(fall, 'all')


def sample_doe(MINX, MAXX, DIMENSION, DOESIZE, OBJECTIVE_FUNCTION):
    X = []
    Y = []
    for i in range(DOESIZE):
        x = [random.uniform(MINX, MAXX) for _ in range(DIMENSION)]
        y = OBJECTIVE_FUNCTION(x)
        X.append(x)
        Y.append(y)
    colours = compute_colours_2(Y)
    return X, Y, colours


def compute_colours_2(Y):
    colours = []
    y_copy = Y.copy()
    y_copy.sort()
    min_value = y_copy[0]
    k = int(0.4 * len(Y))
    m = math.log(0.5) / (y_copy[k] - min_value)
    jet_cmap = mpl.cm.get_cmap(name='jet')
    for y in Y:
        colours.append(jet_cmap(1. - math.exp(m * (y - min_value))))
    return colours


if __name__ == '__main__':
    random.seed(MY_SEED)
    np.random.seed(MY_SEED)
    reproduce_pattern()

