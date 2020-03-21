# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:48:14 2017

@author: Hao Wang
@email: wangronin@gmail.com
"""
from __future__ import print_function
from pdb import set_trace

import pandas as pd
import numpy as np

from numpy import std, array, atleast_2d

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble.base import _partition_estimators
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import r2_score

from joblib import Parallel, delayed

# TODO: implement multi-output/objetive surrogate models, better to model the c
# orrelation among targets
class SurrogateAggregation(object):
    def __init__(self, surrogates, aggregation='WS', **kwargs):
        self.surrogates = surrogates
        self.N = len(self.surrogates)
        self.aggregation = aggregation
        self.weights = np.asarray(kwargs['weights'], dtype='float').ravel()

        assert self.aggregation in ['WS', 'Tchebycheff']

    def fit(self, X, y):
        pass

    def predict(self, X, eval_MSE=False):
        if eval_MSE:
            y_hat_, MSE_ = list(zip(*[s.predict(X, eval_MSE=True) for s in self.surrogates]))
            y_hat_ = np.atleast_2d([_.ravel() for _ in y_hat_])
            MSE_ = np.atleast_2d([_.ravel() for _ in MSE_])
        else:
            y_hat_ = np.atleast_2d([_.predict(X, eval_MSE=False).ravel() for _ in self.surrogates])

        if self.aggregation == 'WS':
            y_hat = self.weights.dot(y_hat_)
            if eval_MSE:
                MSE = (self.weights ** 2.).dot(MSE_)

        elif self.aggregation == 'Tchebycheff':
            # TODO: implement this part
            pass
        
        return (y_hat, MSE) if eval_MSE else y_hat
    
    def gradient(self, X):
        # TODO: implement
        pass
        
# this function has to be globally visible
def save(predict, X, index, out):
    out[:, index] = predict(X, check_input=False)

class RandomForest(RandomForestRegressor):
    """
    Extension on the sklearn RandomForestRegressor class
    Added functionality: empirical MSE of predictions
    """
    def __init__(self, n_estimators=100, max_features=5./6, min_samples_leaf=2, 
                 levels=None, **kwargs):
        """
        parameter
        ---------
        levels : dict, for categorical inputs
            keys: indices of categorical variables
            values: list of levels of categorical variables
        """
        super(RandomForest, self).__init__(n_estimators=n_estimators,
                                           max_features=max_features,
                                           min_samples_leaf=min_samples_leaf,
                                           **kwargs)

        # TODO: using such encoding, feature number will increase drastically
        # TODO: investigate the upper bound (in the sense of cpu time)
        # for categorical levels/variable number
        # in the future, maybe implement binary/multi-value split
        if levels is not None:
            assert isinstance(levels, dict)
            self._levels = levels
            self._cat_idx = list(self._levels.keys())
            self._categories = [list(l) for l in self._levels.values()]

            # encode categorical variables to binary values
            self._enc = OneHotEncoder(categories=self._categories, sparse=False)

    def _check_X(self, X):
        # TODO: this line seems to cause problem sometimes
        X_ = array(X, dtype=object)
        if hasattr(self, '_levels'):
            X_cat = X_[:, self._cat_idx]
            try:
                X_cat = self._enc.transform(X_cat)
            except:
                X_cat = self._enc.fit_transform(X_cat)
            X = np.c_[np.delete(X_, self._cat_idx, 1).astype(float), X_cat]
        return X

    def fit(self, X, y):
        X = self._check_X(X)
        y = y.ravel()
        self.y = y
        return super(RandomForest, self).fit(X, y)

    def predict(self, X, eval_MSE=False):
        # Check data
        X = self._check_X(X)
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
            delayed(save)(e.predict, X, i, y_hat_all) for i, e in enumerate(self.estimators_))

        y_hat = np.mean(y_hat_all, axis=1).flatten()
        if eval_MSE:
            sigma2 = np.std(y_hat_all, axis=1, ddof=1) ** 2.
            sigma2 = sigma2.flatten()
        return (y_hat, sigma2) if eval_MSE else y_hat

# TODO: find a way to wrap R randomforest package
if 11 < 2:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import r, pandas2ri, numpy2ri

    # numpy and pandas data type conversion to R
    numpy2ri.activate()
    pandas2ri.activate()

    class RrandomForest(object):
        """
        Python wrapper for the R 'randomForest' library for regression
        TODO: verify R randomForest uses CART trees instead of C45...
        """
        def __init__(self, levels=None, n_estimators=10, max_features='auto',
                    min_samples_leaf=1, max_leaf_nodes=None, importance=False,
                    nPerm=1, corr_bias=False, seed=None):
            """
            parameter
            ---------
            levels : dict
                dict keys: indices of categorical variables
                dict values: list of levels of categorical variables
            seed : int, random seed
            """
            if max_leaf_nodes is None:
                max_leaf_nodes = ro.NULL

            if max_features == 'auto':
                mtry = 'p'
            elif max_features == 'sqrt':
                mtry = 'int(np.sqrt(p))'
            elif max_features == 'log':
                mtry = 'int(np.log2(p))'
            else:
                mtry = max_features

            self.pkg = importr('randomForest')
            self._levels = levels
            self.param = {'ntree' : int(n_estimators),
                        'mtry' : mtry,
                        'nodesize' : int(min_samples_leaf),
                        'maxnodes' : max_leaf_nodes,
                        'importance' : importance,
                        'nPerm' : int(nPerm),
                        'corr_bias' : corr_bias}

            # make R code reproducible
            if seed is not None:
                r['set.seed'](seed)

        def _check_X(self, X):
            """
            Convert all input types to R data.frame
            """
            if isinstance(X, list):
                if isinstance(X[0], list):
                    X = array(X, dtype=object)
                else:
                    X = array([X], dtype=object)
            elif isinstance(X, np.ndarray):
                if hasattr(self, 'columns'):
                    if X.shape[1] != len(self.columns):
                        X = X.T
            elif isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
                X = X.values

            # be carefull: categorical columns should be converted as FactorVector
            to_r = lambda index, column: ro.FloatVector(column) if index not in self._levels.keys() else \
                ro.FactorVector(column, levels=ro.StrVector(self._levels[index]))
            d = {'X' + str(i) : to_r(i, X[:, i]) for i in range(X.shape[1])}
            X_r = ro.DataFrame(d)

            return X_r

        def fit(self, X, y):
            self.X = self._check_X(X)
            y = array(y).astype(float)

            self.columns = numpy2ri.ri2py(self.X.colnames)
            n_sample, self.n_feature = self.X.nrow, self.X.ncol

            if isinstance(self.param['mtry'], str):
                p = self.n_feature
                self.param['mtry'] = eval(self.param['mtry'])

            self.rf = self.pkg.randomForest(x=self.X, y=y, **self.param)
            return self

        def predict(self, X, eval_MSE=False):
            """
            X should be a dataframe
            """
            X = self._check_X(X)
            _ = self.pkg.predict_randomForest(self.rf, X, predict_all=eval_MSE)

            if eval_MSE:
                y_hat = numpy2ri.ri2py(_[0])
                mse = std(numpy2ri.ri2py(_[1]), axis=1, ddof=1) ** 2.
                return y_hat, mse
            else:
                return numpy2ri.ri2py(_)

if __name__ == '__main__':
    # TODO: this part goes into test 
    # simple test for mixed variables...
    np.random.seed(12)

    n_sample = 110
    levels = ['OK', 'A', 'B', 'C', 'D', 'E']
    X = np.c_[np.random.randn(n_sample, 2).astype(object),
              np.random.choice(levels, size=(n_sample, 1))]
    y = np.sum(X[:, 0:-1] ** 2., axis=1) + 5 * (X[:, -1] == 'OK')

    X_train, y_train = X[:100, :], y[:100]
    X_test, y_test = X[100:, :], y[100:]

    # sklearn-random forest
    rf = RandomForest(levels={2: levels}, max_features='sqrt')
    rf.fit(X_train, y_train)
    y_hat, mse = rf.predict(X_test, eval_MSE=True)

    print('sklearn random forest:')
    print('target :', y_test)
    print('predicted:', y_hat)
    print('MSE:', mse)
    print('r2:', r2_score(y_test, y_hat))
    print()

    if 11 < 2:
        X = np.c_[np.random.randn(n_sample, 2).astype(object),
                  np.random.choice(levels, size=(n_sample, 1))]
        y = np.sum(X[:, 0:-1] ** 2., axis=1) + 5 * (X[:, -1] == 'OK')

        X_train, y_train = X[:100, :], y[:100]
        X_test, y_test = X[100:, :], y[100:]

        rf_ = RandomForest(levels={2: levels}, max_features='sqrt')
        rf_.fit(X_train, y_train)

        rf_aggr = SurrogateAggregation((rf, rf_), weights=(0.1, 0.9))
        y_hat, mse = rf_aggr.predict(X_test, eval_MSE=True)

        print('sklearn random forest:')
        print('target :', y_test)
        print('predicted:', y_hat)
        print('MSE:', mse)
        print('r2:', r2_score(y_test, y_hat))
        print()

    if 11 < 2:
        # R randomForest
        rf = RrandomForest(levels={2: levels}, seed=1, max_features='sqrt')
        rf.fit(X_train, y_train)
        y_hat, mse = rf.predict(X_test, eval_MSE=True)

        print('R randomForest:')
        print('target :', y_test)
        print('predicted:', y_hat)
        print('MSE:', mse)
        print('r2:', r2_score(y_test, y_hat))

    # TODO: those settings should be in test file as inputs to surroagtes
    # leaf_size = max(1, int(n_sample / 20.))
    # ntree=100,
    # mtry=ceil(self.n_feature * 5 / 6.),
    # nodesize=leaf_size
