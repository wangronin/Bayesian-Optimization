import sys

sys.path.insert(0, "../")
import numpy as np
from bayes_optim.surrogate import RandomForest, SurrogateAggregation
from sklearn.metrics import r2_score

np.random.seed(12)


def test_pickling():
    n_sample = 110
    levels = ["OK", "A", "B", "C", "D", "E"]
    X = np.c_[
        np.random.randn(n_sample, 2).astype(object), np.random.choice(levels, size=(n_sample, 1))
    ]
    y = np.sum(X[:, 0:-1] ** 2.0, axis=1) + 5 * (X[:, -1] == "OK")

    X_train, y_train = X[:100, :], y[:100]
    X_test, y_test = X[100:, :], y[100:]

    # sklearn-random forest
    rf = RandomForest(levels={2: levels}, max_features="sqrt")
    rf.fit(X_train, y_train)
    y_hat, mse = rf.predict(X_test, eval_MSE=True)

    print("sklearn random forest:")
    print("target :", y_test)
    print("predicted:", y_hat)
    print("MSE:", mse)
    print("r2:", r2_score(y_test, y_hat))
    print()

    X = np.c_[
        np.random.randn(n_sample, 2).astype(object),
        np.random.choice(levels, size=(n_sample, 1)),
    ]
    y = np.sum(X[:, 0:-1] ** 2.0, axis=1) + 5 * (X[:, -1] == "OK")

    X_train, y_train = X[:100, :], y[:100]
    X_test, y_test = X[100:, :], y[100:]

    rf_ = RandomForest(levels={2: levels}, max_features="sqrt")
    rf_.fit(X_train, y_train)

    rf_aggr = SurrogateAggregation((rf, rf_), weights=(0.1, 0.9))
    y_hat, mse = rf_aggr.predict(X_test, eval_MSE=True)

    print("sklearn random forest:")
    print("target :", y_test)
    print("predicted:", y_hat)
    print("MSE:", mse)
    print("r2:", r2_score(y_test, y_hat))
    print()
