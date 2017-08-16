# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:05:01 2017

@author: wangronin
"""

from __future__ import division   # important! for the float division

import pdb
import sys
import os

import json
import numpy as np
from numpy import array, tile, c_, mean, zeros

import pandas as pd

from regression import regressor
from classification import classifier

# setup working directory: change it to the absolute path on your machine
os.chdir(os.path.expanduser('~')  + '/Dropbox/昊唯分析/Demo')

def preprocessing(df):
    
    # simply filter out the missing values
    column_names = np.array(list(df.columns))
    is_null = array(~df.applymap(pd.isnull).all(0))
    df = df[column_names[array(is_null)]]
    
    df.dropna(axis=0, inplace=True)
    df.dropna(axis=1, inplace=True)
    
    return df
    
    
def modelling(op, algorithm, df, target, selected_feature=None, dropped_feature=None, n_fold=0, 
              parallel=False, verbose=False):
    # select input features and target variable
    feature = set(df.columns) & set(selected_feature) - set(dropped_feature)
    n_feature = len(feature)
    
    X, y = df[feature], df[target]
    if verbose:
        print '{} feature requested: {}'.format(n_sample)
        print 'total feattures: {}'.format(n_feature)
    
    # train the model
    if operation == 'regression':
        model = regressor(algorithm=algorithm)        # our own regression class
    elif operation == 'classification':
        model = classifier(algorithm=algorithm)       # our own classification class
        
    model.fit(X, y, n_fold=10, parallel=parallel)
    
    return model

# -------------------------------------- The Main Engine -------------------------------------------
# get_command_options():
json_file = sys.argv[1]
with open(json_file) as json_data:
    d = json.load(json_data)
    
    # get the request info
    request = d['request']
    operation = request['type']
    verbose = d['verbose']
    
    if operation in ['regression', 'classification']:
        targets = request['regression']
        n_target = len(targets)
        
        dropped_feature = request['dropped_feature']
        selected_feature = request['selected_feature']
        
        algorithm = request['algorithm']
        parallel = request['parallel_execution']
        
    elif operation == 'clustering':
        pass
    
    elif operation == 'imputation':
        pass
    
    # get output request
    output_request = d['output']
    
    # get the data src info and load data
    data_info = d['data_source']
    data_src_type = data_info['type']
    data_conn = data_info['conn']
    
    if data_src_type == 'csv':
        df = pd.read_csv(data_conn)
        
    elif data_src_type == 'sql':
        sql_query = data_conn
        # TODO: complete the implementation here
        pass
    
    n_sample, n_feature = df.shape
    if verbose:
        print 'total records: {}'.format(n_sample)
        print 'total feattures: {}'.format(n_feature)

    # TODO: improve missing data imputation
    # preprocessing
    df = preprocessing(df)
    
    if operation in ['regression', 'classification']:
        # modelling for each target
        for i, t in enumerate(targets):
            if verbose:
                print 'building model for target {}, {}/{}'.format(t, i+1, n_target)
                
            model = modelling(operation, algorithm, df, t, selected_feature, dropped_feature, 
                              parallel=parallel, verbose=verbose)
                              
            for i, c in enumerate(output_request):                     
                # dumpe the model
                command = c['command']
                if command == 'dump_model':
                    model.dump(path=c['file'])
            
    elif operation == 'clustering':
        pass
    
    elif operation == 'imputation':
        pass
    
    
