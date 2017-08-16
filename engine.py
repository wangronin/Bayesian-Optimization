#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:10:05 2017

@author: wangronin
"""

import pdb, sys

import numpy as np
import pandas as pd

from configurator import configurator

# test: Usage
# python engine.py input.json
if len(sys.argv) == 2:
    cmd_file = sys.argv[1]
    import json
    with open(cmd_file) as json_data:
        d = json.load(json_data)

        ID = d['ID']

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

        # get the request info
        request = d['request']
        operation = request['type']
        verbose = d['verbose']

        targets = request['targets']
        n_target = len(targets)

        dropped_feature = request['dropped_feature']
        selected_feature = request['selected_feature']

        all_feature = list(set(df.columns) - set(targets))
        selected_feature = all_feature if len(selected_feature) == 0 else selected_feature
        selected_feature = list(set(selected_feature) - set(dropped_feature))

        X, y = df[selected_feature].as_matrix(), df[targets].as_matrix()
        is_parallel = request['parallel_execution']

        # get output request
        output_request = d['output']
        output_file = output_request['file']

        if operation == 'configuration':
            # to configure a SVM regression model
            par = d['additional_parameters']
            conf = configurator(n_iter=par['max_iter'],
                                algorithm=par['model'],
                                metric=par['metric'],
                                n_fold=par['nfold'],
                                par_list=par['par_list'],
                                par_lb=par['par_lb'],
                                par_ub=par['par_ub'],
                                conn=d['conn'],
                                ID=ID,
                                verbose=verbose)

            res = conf.configure(X, y)

            output_file += '.csv'
            pars = conf._configurator__conf_optimizer.X
            perf = conf._configurator__conf_optimizer.y.reshape(-1, 1)

            df = pd.DataFrame(np.c_[pars, perf], columns=par['par_list'] + ['perf'])
            df.to_csv(output_file, index=False)

        # TODO: implement the rest...
        elif operation == 'clustering':
            pass

        elif operation == 'imputation':
            pass



    sys.exit()
