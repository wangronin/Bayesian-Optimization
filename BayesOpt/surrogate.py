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
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri

from sklearn.ensemble import RandomForestRegressor

# numpy and pandas data type conversion to R
numpy2ri.activate()
pandas2ri.activate()

class RrandomForest(object):
    """
    Python wrapper for the R 'randomForest' library
    """
    def __init__(self):
        self.pkg = importr('randomForest')

    def fit(self, X, y):
        self.columns = X.columns
        self.X = X
        self.n_sample, self.n_feature = X.shape
        # if not isinstance(y, np.ndarray):
        #     y = np.array(y)
        self.rf = self.pkg.randomForest(x=X, y=y, ntree=100,
                                        mtry=ceil(self.n_feature * 5 / 6.),
                                        nodesize=10)
        return self

    def predict(self, X, eval_MSE=False):
        """
        X should be a dataframe
        """
        if isinstance(X, list):
            X = pd.DataFrame([X], columns=self.columns)
        elif isinstance(X, pd.Series):
            X.index = self.columns
            X = pd.DataFrame([X])
        elif isinstance(X, np.ndarray):
            if X.shape[1] != len(self.columns):
                X = X.T
            X = pd.DataFrame(X, columns=self.columns)
        n_sample = X.shape[0]
        X = X.append(self.X) # ad hoc fix for R 'randomForest' package
            
        _ = self.pkg.predict_randomForest(self.rf, X, predict_all=eval_MSE)
        if eval_MSE:
            y_hat = array(_[0])[:n_sample]
            mse = std(atleast_2d(_[1])[0:n_sample, :], axis=1, ddof=1) ** 2
            return y_hat, mse
        else:
            return array(_)[:n_sample]
        
        