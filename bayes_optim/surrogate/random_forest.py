from __future__ import annotations

from collections import OrderedDict
from typing import List, Union

import numpy as np
from joblib import Parallel, delayed
from numpy import array
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._base import _partition_estimators
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from ..solution import Solution

__authors__ = ["Hao Wang"]


class SurrogateAggregation(object):
    """Linear aggregation of surrogate models used for multi-obvjective optimization"""

    def __init__(self, surrogates, aggregation="WS", **kwargs):
        self.surrogates = surrogates
        self.N = len(self.surrogates)
        self.aggregation = aggregation
        self.weights = np.asarray(kwargs["weights"], dtype="float").ravel()

        assert self.aggregation in ["WS", "Tchebycheff"]

    def fit(self, X, y):
        pass

    def predict(self, X, eval_MSE=False):
        if eval_MSE:
            y_hat_, MSE_ = list(zip(*[s.predict(X, eval_MSE=True) for s in self.surrogates]))
            y_hat_ = np.atleast_2d([_.ravel() for _ in y_hat_])
            MSE_ = np.atleast_2d([_.ravel() for _ in MSE_])
        else:
            y_hat_ = np.atleast_2d([_.predict(X, eval_MSE=False).ravel() for _ in self.surrogates])

        if self.aggregation == "WS":
            y_hat = self.weights.dot(y_hat_)
            if eval_MSE:
                MSE = (self.weights ** 2.0).dot(MSE_)

        elif self.aggregation == "Tchebycheff":
            # TODO: implement this part
            pass

        return (y_hat, MSE) if eval_MSE else y_hat

    def gradient(self, X):
        # TODO: this model is not differentiable?
        pass


def _save_prediction(predict, X, index, out):
    """It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    out[..., index] = predict(X, check_input=False)


class RandomForest(RandomForestRegressor):
    r"""Extension on the sklearn's `RandomForestRegressor`
    Added functionality:
        1) MSE estimate,
        2) OneHotEncoding to handle categorical variables
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: float = 5 / 6,
        min_samples_leaf: int = 2,
        levels: dict = None,
        **kwargs,
    ):
        """
        parameter
        ---------
        levels : dict, for categorical inputs
            keys: indices of categorical variables
            values: list of levels of categorical variables
        """
        super().__init__(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            **kwargs,
        )
        assert isinstance(levels, dict)
        self.levels = levels
        self.is_fitted = False

        # for categorical levels/variable number
        # in the future, maybe implement binary/multi-value split
        if self.levels:
            self._levels = OrderedDict(sorted(levels.items()))
            self._cat_idx = list(self._levels.keys())
            self._categories = list(self._levels.values())
            # encode categorical variables to binary values
            self._enc = OneHotEncoder(categories=self._categories, sparse=False)

    def _check_X(self, X: Union[Solution, List, np.ndarray]) -> Solution:
        X_ = array(X, dtype=object)
        if hasattr(self, "_levels"):
            X_cat = X_[:, self._cat_idx]
            try:
                X_cat = self._enc.transform(X_cat)
            except Exception:
                X_cat = self._enc.fit_transform(X_cat)
            X = np.c_[np.delete(X_, self._cat_idx, 1).astype(float), X_cat]
        return X

    def fit(self, X: Union[Solution, List, np.ndarray], y: np.ndarray):
        self.X = self._check_X(X)
        self.y = y
        self.is_fitted = True
        return super().fit(self.X, self.y)

    def predict(self, X: Union[Solution, List, np.ndarray], eval_MSE=False) -> np.ndarray:
        """Predict regression target for `X`.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        # check data
        X = self._check_X(X)
        X = self._validate_X_predict(X)
        # assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        # storing the output of every estimator since those are required to estimate the MSE
        y_hat_all = (
            np.zeros((X.shape[0], self.n_outputs_, self.n_estimators), dtype=np.float64)
            if self.n_outputs_ > 1
            else np.zeros((X.shape[0], self.n_estimators), dtype=np.float64)
        )
        # parallel loop
        Parallel(n_jobs=n_jobs, verbose=self.verbose, backend="threading")(
            delayed(_save_prediction)(e.predict, X, i, y_hat_all)
            for i, e in enumerate(self.estimators_)
        )
        y_hat = np.mean(y_hat_all, axis=-1)

        if eval_MSE:
            # TODO: implement the jackknife estimate of variance
            MSE_hat = np.std(y_hat_all, axis=-1, ddof=1) ** 2.0
        return (y_hat, MSE_hat) if eval_MSE else y_hat
