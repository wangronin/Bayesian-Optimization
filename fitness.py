# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 12:33:38 2012

@author: H.Wang
"""

import pdb
import numpy as np
from numpy import cos, exp, pi, power, arange, subtract, sqrt, array, size


#---------------------------- General Fitness Function Interface -------------------
class fitness(object):
    
    def __init__(self, inputformat='col', *args):
        
        # Indicating the axis along which the computation is performed
        if inputformat == 'col':
            self._axis = 0
        elif inputformat == 'row':
            self.__axis = 1
        else:
            self._axis = 0
            raise ValueError
            
        self.evaluations = 0
        self._nfreefuns = (linear, sphere, schwefel, rastrigin, M, axis_parallel_ellipsoid, cigar, \
            tablet, cigartab, ellipsoid, parabR, sharpR, diffpow, rosenbrock,\
            ackley, powersum, rand, branin)
            
        self.dictnfreefuns = dict((i+1, f) for i, f in enumerate(self._nfreefuns))
        self.nfreeIDs = sorted(self.dictnfreefuns.keys()) 
        self._currentID = None
        self.ftarget = None
    
    def __call__(self, x):
        return self.evalfun(x)
        
    def __repr__(self):
        try:
            S = 'function {}'.format(str(self.dictnfreefuns[self._currentID]).split(' ')[1])
            return S 
        except:
            print "haha!"
        
    def __evaluations(self, x):
        self.evaluations += x.shape[1]
    
    def setfun(self, fID):
        
        if any(fID == array(self.nfreeIDs)):
            self._currentID = fID
        else:
            raise ValueError
    
    def evalfun(self, x):
        
        x = np.atleast_2d(x)
        if self._axis == 1:
            x = x.T
            
        self.__evaluations(x)
        
        if self._currentID is None:
            print 'function ID is not set!'
            raise ValueError
            
        return self.dictnfreefuns[self._currentID](x)
        


#---------------------------- Fitness Functions Implemmentations ------------------
   
# TODO: translation of optimum, perturbation, noisy-version, rotation transformation
offset = None
rotation = None

def M(x, alpha=6):
    dim = x.shape[0]
    return -np.sum(np.sin(5*pi*x)**alpha, axis=0) / dim + 11
    
def const(x):
    n = x.shape[1]
    return np.repeat(2, n)
    
def rand(x):
    
    n = x.shape[1]
    return np.random.rand(n)
    
def linear(x):
    return x[0, :]
    
def sphere(x):
    return np.sum(power(x, 2), axis=0) 
    
def schwefel(x):
#    global offset, rotation
#    n = x.shape[0]
#    if offset is None or np.size(offset, 0) != n:
#        offset = np.random.rand(n, 1)
#    if rotation is None or np.size(rotation, 0) != n:
#        rotation = np.linalg.qr(np.random.randn(n, n))[0]
#    x = np.dot(rotation, x) + offset
    cum = np.cumsum(x, axis=0)
    p = power(cum, 2)
    return np.sum(p, axis=0)

def rastrigin(x):
    n = np.size(x, 0)
#    x = x - array([0.6, 2.3]).reshape(-1, 1)
    n = np.size(x, 0)
    res = np.sum(x**2 - 10*np.cos(2*pi*x), axis=0)
    return res + 10*n
    
def axis_parallel_ellipsoid(x):

    n = x.shape[0]
    t = x * (np.arange(1, n+1)).reshape(-1, 1)
    p = power(t, 2)
    return np.sum(p, axis=0)
    
def cigar(x):
    
    f = x[0, :]**2 + 1e6*np.sum(x[1::, :]**2, axis=0)
    return f
    
def tablet(x):
    
    f = 1e6*x[0, :]**2 + np.sum(x[1::, :]**2, axis=0)
    return f
    
def cigartab(x):
    
    f = x[0, :]**2 + 1e8*x[-1, :]**2 + 1e4*np.sum(x[1:-1, :]**2, axis=0)
    return f
    
def ellipsoid(x):
    
    N = x.shape[0]
    co = array([1e3**(i/(N-1.0)) for i in arange(0, N)]).reshape(-1, 1)
    f = np.sum((co*x)**2, axis=0)
    return f
    
def parabR(x):

    f = -x[0, :] + 100*np.sum(x[1:, :]**2, axis=0)
    return f

def sharpR(x):
    
    f = -x[0, :] + 100*sqrt(np.sum(x[1:, :]**2, axis=0))
    return f
    
def diffpow(x):
    
    N = x.shape[0]
    x = np.abs(x)
    p = (10*arange(0, N) / (N-1.0) + 2).reshape(-1, 1)
    f = np.sum(power(x, p), axis=0)
    return f

def rosenbrock(x):
   
    sqx = power(x, 2)
    foobar = power(subtract(x[0:-1, :], 1), 2)
    diff = subtract(sqx[0:-1, :], x[1::, :])
    term = 100*power(diff, 2) + foobar
    return np.sum(term, axis=0)    

def powersum(x):
    
    N = x.shape[0]
    t = np.abs(x)
    p = power(t, np.arange(2, N+2).reshape(-1, 1))
    return np.sum(p, axis=0)

def ackley(x):

    c1 = -20
    c2 = -0.2
    c3 = 2*pi
    foobar = c1*exp(c2*np.sqrt(np.mean(power(x, 2), axis=0)))
    res = foobar - exp(np.mean(cos(c3*x), axis=0)) - c1 + exp(1)
    return res
    
    
def branin(x):
    
    x = np.atleast_2d(x)

    x = x.T if x.shape[0] != 2 else x
    
    x1 = x[0, :]
    x2 = x[1, :]
    y = (x2 - (5.1/(4*pi**2))*x1**2 + 5*x1/pi-6)**2 + 10*(1-1./(8*pi))*cos(x1) + 10
    return  y
    
def step(x):
    
    return np.sum((np.floor(x + 0.5))**2.0, axis=0)
    

def himmelblau(x):
    x0, x1 = x
    
    return (x0 ** 2. + x1 - 11.) ** 2. + (x0 + x1 ** 2. - 7) ** 2.
    

def hartman6(x):
    
    x = x.T if size(x, 1) != 6 else x
    n_sample = size(x, 0)
    
    alpha = array([1.0, 1.2, 3.0, 3.2])
    A = array([[10, 3, 17, 3.5, 1.7, 8],
               [0.05, 10, 17, 0.1, 8, 14],
               [3, 3.5, 1.7, 10, 17, 8],
               [17, 8, 0.05, 10, 0.1, 14]])
    P = 10**(-4.0) * array([[1312, 1696, 5569, 124, 8283, 5886],
                            [2329, 4135, 8307, 3736, 1004, 9991],
                            [2348, 1451, 3522, 2883, 3047, 6650],
                            [047, 8828, 8732, 5743, 1091, 381]])
    
    res = zeros(n_sample)
    for i in range(n_sample):
        tmp = np.dot(A, ((x[i, :] - P)**2.0).T)
        res[i] = -np.dot(alpha, np.exp(-array([tmp[0, 0], tmp[1, 1], tmp[2, 2], tmp[3, 3]])))
        
    return res
    


#def weierstrass(x, a, b):
#    
#    l = x.shape[0]
#    critial_value = 1e-10
#    n = nextpow(1.0/critial_value, 1/a)
#    res = zeros(l)
#    
#    for i in range(n):
#        res += a**i * cos(b**i * pi*x)
#        
#    return res
#
#def nextpow(i, base=2):
#    n = base
#    c = 1
#    while n < i: 
#        n = n * base
#        c += 1
#    return c

    




    

    

    