#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:48:14 2017

@author: wangronin
"""

import pdb
import numpy as np
from numpy import ceil, std, array, atleast_2d
import pandas as pd

from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri, numpy2ri

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble.base import _partition_estimators
from joblib import Parallel, delayed

# numpy and pandas data type conversion to R
numpy2ri.activate()
pandas2ri.activate()

def save_prediction(predict, X, index, out):
    out[:, index] = predict(X, check_input=False)

# TODO: add support for categorical data
class RandomForest(RandomForestRegressor):
    """
    Extension for the sklearn RandomForestRegressor class
    Added functionality: empirical MSE of predictions
    """
    def predict(self, X, eval_MSE=False):
        check_is_fitted(self, 'estimators_')
        # Check data
        X = np.atleast_2d(X)
        X = self._validate_X_predict(X)
        
        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat_all = np.zeros((X.shape[0], self.n_outputs_, self.n_estimators), dtype=np.float64)
        else:
            y_hat_all = np.zeros((X.shape[0], self.n_estimators), dtype=np.float64)

        # Parallel loop
        Parallel(n_jobs=n_jobs, verbose=self.verbose, backend="threading")(
            delayed(save_prediction)(e.predict, X, i, y_hat_all) for i, e in enumerate(self.estimators_))

        y_hat = np.mean(y_hat_all, axis=1).flatten()
        if eval_MSE:
            sigma2 = np.std(y_hat_all, axis=1, ddof=1) ** 2.
            sigma2 = sigma2.flatten()
        return (y_hat, sigma2) if eval_MSE else y_hat

class RrandomForest(object):
    """
    Python wrapper for the R 'randomForest' library
    """
    def __init__(self, seed=None):
        self.pkg = importr('randomForest')

        # TODO: make R code reproducible, failed...
        if seed is not None:
            r['set.seed'](seed)
        
    def _check_X(self, X):
        """
        Convert all input types to pandas dataframe
        """
        if isinstance(X, list):
            if isinstance(X[0], list):
                X = pd.DataFrame(X)
            else:
                X = pd.DataFrame([X])

        elif isinstance(X, pd.Series):
            X.index = pd.RangeIndex(0, len(X))
            X = pd.DataFrame([X])

        elif isinstance(X, np.ndarray):
            if hasattr(self, 'columns'):
                if X.shape[1] != len(self.columns):
                    X = X.T
            X = pd.DataFrame(X)
        return X

    def fit(self, X, y):
        self.X = self._check_X(X)
        self.columns = self.X.columns
        n_sample, self.n_feature = self.X.shape
        leaf_size = max(1, int(n_sample / 20.))
        
        self.rf = self.pkg.randomForest(x=self.X, y=y, ntree=100,
                                        mtry=ceil(self.n_feature * 5 / 6.),
                                        nodesize=leaf_size)
        return self

    def predict(self, X, eval_MSE=False):
        """
        X should be a dataframe
        """
        X = self._check_X(X)
        n_sample = X.shape[0]
        
        # ad hoc fix for R 'randomForest' package
        X = X.append(self.X)
        X.reset_index(drop=True, inplace=True)
            
        _ = self.pkg.predict_randomForest(self.rf, X, predict_all=eval_MSE)
        if eval_MSE:
            y_hat = array(_[0])[:n_sample]
            mse = std(atleast_2d(_[1])[0:n_sample, :], axis=1, ddof=1) ** 2.
            return y_hat, mse
        else:
            return array(_)[:n_sample]

if __name__ == '__main__':
    X = np.random.randn(100, 2)
    y = np.sum(X ** 2., axis=1)

    # sklearn-random forest
    rf = RandomForest()
    rf.fit(X, y)

    print rf.predict(X[:2, ], eval_MSE=True)
    
    # R randomForest
    rf = RrandomForest()
    rf.fit(X, y)

    print rf.predict(X[:2, ], eval_MSE=True)
        
        