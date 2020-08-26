# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 17:03:43 2014

@author: wangronin
"""

from pdb import set_trace
import numpy as np

from ..misc import boundary_handling
from numpy.linalg import norm, cholesky, LinAlgError
from numpy.random import randn, rand
from numpy import sqrt, eye, exp, dot, inf, zeros, outer, triu, isinf, isreal, ones

# TODO: implement this in C++
class one_plus_one_cma_es(object):
    def __init__(
        self, 
        dim, 
        obj_func,
        x0,
        sigma0, 
        max_FEs,
        lb = None,
        ub = None,
        opts = {}
        ):
        
        self.parent = eval(x0) if isinstance(x0, str) else x0 
        self.eval_budget = int(eval(max_FEs)) if isinstance(max_FEs, str) \
            else int(max_FEs)
        
        self.dim = dim
        self.sigma_init = sigma0
        self.sigma = sigma0
        self.f_target = opts['f_target']
        self.lb = lb if lb is not None else None
        self.ub = ub if ub is not None else None
        self.fitness = obj_func
        self.is_boundary_handling = True
        self.is_cholesky = True
    
        # Exogenous strategy parameters 
        self.p_threshold = 0.44
        self.p_succ_target = 2. / 11.
        self.p_succ = self.p_succ_target
        self.c_p = 1. / 12.
        self.ccov = 2. / (dim ** 2 + 6.)
        self.d = 1.0 + dim / 2.0
        
        if self.is_cholesky:
            self.A = eye(dim)
            self.c_a = sqrt(1 - self.ccov)
        else:
            self.C = eye(dim)
            self.A = eye(dim)
            self.pc = zeros((dim, 1))
            self.cc = 2. / (dim + 2.)
        
        # Parameters for evolution loop
        self.eval_count = 0
        self.xopt = self.parent
        self.fopt, self.f_parent = inf, inf
        self.exception_info = 0
        self.stop_list = []

    def run(self):
        # Evolution loop       
        while len(self.stop_list) == 0:
            self.step()
            self.exception_handle()
            self.restart_criteria()
            
        return self.xopt, self.fopt, self.eval_count, self.stop_list
            
    def step(self):
        offspring, z = self.mutate()
        
        # Evaluation
        f_offspring = self.fitness(offspring)
        self.eval_count += 1
        
        # selection
        is_success = f_offspring < self.f_parent
        
        # Parameter adaptation
        self.p_succ = (1 - self.c_p) * self.p_succ + self.c_p * is_success
        self.update_step_size(is_success)

        if is_success:
            self.f_parent = self.fopt = f_offspring
            self.parent = self.xopt = offspring
             
            if self.is_cholesky:
                self.update_cov_cholesky(z)
            else:
                self.update_cov(z)
                
                # Cholesky decomposition
                if np.any(isinf(self.C)):
                    self.exception_info = 1
                else:
                    try:
                        A = cholesky(self.C)
                        if np.any(~isreal(A)):
                            self.exception_info = 1
                        else:
                            self.A = A
                    except LinAlgError:
                        self.exception_info = 1
        
    def mutate(self):
        z = randn(self.dim, 1)
        offspring = self.parent + self.sigma * dot(self.A, z)
        if self.is_boundary_handling:
            offspring = boundary_handling(offspring, self.lb, self.ub) 
            
        return offspring, z
                            
    def exception_handle(self):
        """Handling warings: Internally rectification of strategy parameters
        """
        if self.sigma < 1e-16 or self.sigma > 1e16:
            self.exception_info = 1
        
        if self.exception_info != 0:
            if not self.is_cholesky:
                self.C = eye(self.dim)
                self.pc = zeros((self.dim, 1))
            self.A = eye(self.dim)
            self.sigma = self.sigma_init
            self.exception_info = 0
            
    def update_step_size(self, is_success):
        self.sigma *= exp(
            (self.p_succ - self.p_succ_target) / ((1 - self.p_succ_target) * self.d)
        )
        
    def update_cov(self, z):
        cc = self.cc
        ccov = self.ccov
        if self.p_succ < self.p_threshold:
            self.pc = (1 - cc) * self.pc + sqrt(cc * (2-cc)) * dot(self.A, z)
            self.C = (1 - ccov) * self.C + ccov * outer(self.pc, self.pc)
        else:
            self.pc = (1 - cc) * self.pc
            self.C = (1 - ccov) * self.C + ccov * (
                outer(self.pc, self.pc) + cc*(2-cc) * self.C
            )
        self.C = triu(self.C) + triu(self.C, 1).T 
    
    def update_cov_cholesky(self, z):
        if self.p_succ < self.p_threshold:
            c_a = self.c_a
            coeff = c_a * (sqrt(1 + (1-c_a ** 2) * norm(z) ** 2 / c_a ** 2) - 1) / norm(z) ** 2
            self.A = c_a * self.A + coeff * dot(dot(self.A, z), z.T)
    
    def restart_criteria(self):
        if self.fopt <= self.f_target:
            self.stop_list.append('ftarget')
            
        if self.eval_count >= self.eval_budget:
            self.stop_list.append('maxFEs')