#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:35:30 2019

@author: wangronin
"""

class ConstrainedBO(BO):
    def __init__(self, constraint_func, *argv, **kwargs):
        super(ConstrainedBO, self).__init__(*argv, **kwargs)
        self.constraint_func = constraint_func
        assert hasattr(self.constraint_func, '__call__')

    def evaluate(self):
        pass

    def fit_and_assess(self):
        pass

    