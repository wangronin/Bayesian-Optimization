# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 20:24:50 2013

@author: Wang Hao
"""

import pdb
import time
import numpy as np
from multiprocessing import Process, Array, Lock
from scipy.stats import gaussian_kde, chi
from scipy.integrate import quad, simps
from numpy.random import rand, randn
from numpy.linalg import norm, lstsq, qr
from numpy import pi, eye, cos, sin, dot, arange, floor, sort, power, linspace, log,\
    append, exp, sqrt, ceil, size, array, inf, inner, zeros, outer, empty, shape
from ctypes import cdll, c_int, c_uint, c_double, POINTER

#import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
#from matplotlib import cm

#c_noisy_library = cdll.LoadLibrary('/home/wangronin/Desktop/ES/libnoisy.so')
#c_noisy_gen = c_noisy_library.nois_gen
#c_noisy_gen.argtypes = [c_uint, POINTER(c_uint), c_uint, POINTER(c_double),\
#    POINTER(c_double)]
#c_noisy_gen.restypes = c_int


def level_cal(Z, step=0.2):
    """
    
    Calculate the levels of the contour lines in log scale
    """
    base = 10
    _min, _max = min(Z.flatten()), max(Z.flatten())
    levels = base**(arange(log(1e-30)/log(base), log(_max-_min+1e-30)/log(base), step)) + _min - 1e-30
    
    pdb.set_trace()

    return levels
    
def colors_cal(levels):
    """
    
    Calculate the colors due to log scaled value
    """
    cNorm = LogNorm(vmin=1e-2, vmax=levels.max()-levels.min() + 1e-2)
    smap = cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('jet'))
    colors = smap.to_rgba(levels-levels.min())
    return colors
    
def gram_schmidt(V, C=None):
    """
    
    Modified Gram-Schmidt process
    Compute the conjugate direction set with respect to C
    """
    U = np.empty(V.shape)
    n = V.shape[1]
    
    if C == None:
        
        for i in range(n):
            u = V[:, i]
            for j in range(i):
                u -= inner(u, U[:, j]) * U[:, j]
            U[:, i] = u / norm(u)
            
    else:
        
        for i in range(n):
            u = V[:, i]
            for j in range(i):
                u -= inner(u, dot(C, U[:, j])) * U[:, j] / inner(U[:, j], dot(C, U[:, j]))
            U[:, i] = u / norm(u)
    
    return U
    
def measure(s):
    N = s.shape[1]
    m = 0.0
    for i in range(N-1):
        for j in range(i+1, N):
            m += dot(s[:, i], s[:, j]) / (norm(s[:, i])*norm(s[:, j]))
    m /= N*(N-1)/2.
    return m

def rand_orth_mat(dim, method=3):
    """ 
    Generate a rotation operator (matrix) as a noise 
    default generation method is to use C(GSL) code 
    """
    
#    angle = pi / sqrt(dim/4.0)
#    pm = .2
#    angle = pi*np.sqrt(dim*8.) / 30.
    angle = pi / 4.
    pm = 1
    
    operator = eye(dim)
    R = eye(dim)
    
    if method == 0:
        for k in arange(0, dim-1):
            for j in arange(k+1, dim):
                if rand() < pm:
                    rotation = angle * randn()/3.0
                    R[k, k] = cos(rotation)
                    R[j, j] = R[k, k]
                    R[k, j] = -sin(rotation)
                    R[j, k] = -R[k, j]    
                    operator = dot(R, operator)
                    R[k, k] = 1
                    R[j, j] = 1
                    R[k, j] = 0
                    R[j, k] = 0     
                    
    elif method == 1:
        m = 0
        n_max = dim*(dim-1)/2
        index = (rand(n_max) < pm) * arange(1, n_max+1)
        index = index[index > 0]
        index = index - 1
        
        n_rotation = len(index) 
        rotations = angle * (2*rand(n_rotation) - 1)   
        cum = np.cumsum(arange(dim-1, 0, -1))
        for r in index:
            foobar = len(cum[(cum - r-1) >= 0])
            k = dim-1 - foobar
            j = int(r+1 - k*(dim-(k+1)/2.0) + k)
            
            R[k, k] = cos(rotations[m])
            R[j, j] = R[k, k]
            R[k, j] = -sin(rotations[m])
            R[j, k] = -R[k, j]
    
            operator = dot(R, operator)
    
            R[k, k] = 1
            R[j, j] = 1
            R[k, j] = 0
            R[j, k] = 0
            m += 1
            
    elif method == 3:
        # The wrapper procedure  for 'c_noisy_gen' implemented in C (GSL)
        n_max = dim*(dim-1)/2
    
        index = (rand(n_max) < pm) * arange(1, n_max+1)
        index = index[index > 0]
        index = index - 1
        
        n_rotation = index.shape[0]
        rotations = angle * (2*rand(n_rotation) - 1)
#        res = zeros((dim, dim))
        
        # Prepare the arguments for C function
        POS = index.ctypes.data_as(POINTER(c_uint))
        ROTATIONS = rotations.ctypes.data_as(POINTER(c_double))
        RES = operator.ctypes.data_as(POINTER(c_double))
        
        # Invoke the C code for dense matrix multiplication
        c_noisy_gen(dim, POS, n_rotation, ROTATIONS, RES)
    
    # Subgroup algorithm
    elif method == 4:
        for i in arange(2, dim+1):
            if i == 2:
                theta = 2*pi*rand()
                minor = array([[cos(theta), -sin(theta)],
                               [sin(theta), cos(theta)]])
            else:
                v = randn(i, 1)
                v = v / norm(v)
                e1 = zeros((i, 1))
                e1[0] = 1
                x = (e1 - v) / norm(e1 - v)
                H = eye(i) - 2*outer(x, x)
                minor = append(zeros((1, i-1)), minor, axis=0)
                minor = append(e1, minor, axis=1)
                minor = dot(H, minor)
                
        operator = minor
    return operator
    

#def noise_gen_c(dim):
#    """ 
#    The wrapper fucntion for 'c_noisy_gen'
#    Do the same work as 'noise_gen' however, it is implemented in C (GSL)
#    """
#    
#    angle = c_double(pi/100)
#    pm = c_double(.1)
#    op_mat = zeros((dim, dim))
#    P_OP_MAT = op_mat.ctypes.data_as(POINTER(c_double))
#    c_noisy_gen(dim, pm, angle, P_OP_MAT)
#    
#    return op_mat 

def rotation_matrix(x, y):
    """
    Calculate the rotation matrix between two high dimensional vectors
    """
    dim = size(x)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    x = x / norm(x)
    y = y / norm(y)
    u = x - y 
    u = u / norm(u)
    R = eye(dim) -  2*np.outer(u, u)
    
    A = append(x.T, y.T, axis=0)
    B = array([[0], [1]])
    c = lstsq(A, B)[0]
    c = c / norm(c)
    S = eye(dim) - 2*np.outer(c, c)
    
    return dot(R, S)

    
def qqplot(x, y=None):
    """
    Quantile-Quantile Plot function to compare two data sets or
    compare one data set to the specified theoretical distribution
    """
    import matplotlib.pyplot as plt
    if y == None:      # quantiles of data against gaussian distribution
        x = sort(x)
        n = x.shape[0]
        pp = arange(n) / (n*1.0)
        q = stats.norm.ppf(pp)
        plt.plot(q, x, 'r+')
        plt.xlabel('Theoretical Normal distribution')
        plt.ylabel('X data')
        plt.title('Q-Q plot')
        plt.show()    
    else:             # quantiles of X against that of Y
        n_x = x.shape[0]
        n_y = y.shape[0]
        n = min(n_x, n_y)
        pp = arange(n) / (n*1.0)
        x = sort(x)
        y = sort(y)
        iX = floor(pp*n_x).astype(int)
        iY = floor(pp*n_y).astype(int)
        plt.plot(x[iX], y[iY], 'r.')
        plt.xlabel('X Data')
        plt.ylabel('Y data')
        plt.title('Q-Q plot')
        plt.show()
        
def unit_circle(p):
    """
    Show the 2-D unit circle in p-norm space
    """
    import matplotlib.pyplot as plt
    dx = .01
    x = append(arange(-1, 1, dx), 1)
    y1 = power(1 - abs(x)**p, 1.0/p)
    y2 = -y1
    plt.figure(figsize=(8, 7.5))
    plt.plot(x, y1, 'r', x, y2, 'r')
    plt.xlim(x[0]-3*dx, x[-1]+3*dx)
    plt.ylim(max(y1)+3*dx, -max(y1)-3*dx)
    plt.title('Unit Circle in $\ell^{' + str(p) + '}$ space')
    plt.grid(True)
    plt.show()

def test(dim, mu):
    tau = 1. / sqrt(dim)
    a = exp(tau**2 / 2.)
    b = a**2
    res = (b*(b-1) + mu*(a-1)**2) / (2*mu*(a-1))
    return res
    
        

def c_1_lambda(m):
    """
    Numerical computation for 
    progress coefficient c(1, lambda)
    """
    from scipy.stats import norm
    pdf, cdf = norm.pdf, norm.cdf
    
    i = lambda x: x * m * pdf(x)*cdf(x)**(m-1.)
    E, err = quad(i, -inf, inf)
    return E
    
def c_1_lambda_m(m):
    """
    Numerical computation for 
    progress coefficient c(1, lambda_m)
    
    """
    from scipy.stats import norm
    pdf, cdf = norm.pdf, norm.cdf
    
    m = floor(m/2)
    i = lambda x: x * 2 * m * pdf(x) * (cdf(x)-cdf(-x))**(m-1.)
    E, err = quad(i, 0, inf)
    return E
    
def __c_1_lambda_om(_lambda, dim):
    """
    Numerical computation for 
    progress coefficient c(1, lambda_om)
    
    """
    
    def __c_1_lambda_om(_lambda, dim, proj_max):
       
        _lambda_half = int(ceil(_lambda / 2.0))
        
        # for memory concern
        n_sample = len(proj_max)
        
        for i in range(n_sample):
            
            # mirrored orthogonal sampling
            q = qr(randn(dim, dim))[0]
            l = chi.rvs(dim, size=dim)
            s = l * q
            samples = s[:, 0:_lambda_half]
            samples = append(samples, -samples, axis=1)
            
            # projection onto e1
            proj = samples[0, :]
            
            # the largest order statistic
            proj_sorted = sort(proj)
            proj_max[i] = proj_sorted[-1]
    
    
    n_trial = 1e5
    n_worker = 4
    n_sample = n_trial / n_worker
    
    proj_max = [Array('f', zeros(n_sample)) for i in range(n_worker)]
    procs = [Process(target=__c_1_lambda_om, args=[_lambda, dim, proj_max[i]]) for i in range(n_worker)]
        
    # Starting the parallel computation
    for p in procs: p.start()
    for p in procs: p.join()
    
    all_sample = empty(n_trial)
    i = 0
    for a in proj_max:
        all_sample[i:i+len(a)] = array(a)
        i += len(a)
    
    epdf_om = gaussian_kde(all_sample)
    
    i = lambda x: x * epdf_om(x)
    E, err = quad(i, -inf, inf)
    
    return E
    
def c_1_lambda_om(_lambda, dim):
    """
    Numerical computation for 
    progress coefficient c(1, lambda_om)
    
    """
    
    def __c_1_lambda_om(_lambda, dim, ecdf):
       
        _lambda_half = int(ceil(_lambda / 2.0))
        
        proj_max = zeros(n_sample_per_cycle)
        
        for i in range(n_sample):
            
            # mirrored orthogonal sampling
            q = qr(randn(dim, dim))[0]
            l = chi.rvs(dim, size=dim)
            s = l * q
            samples = s[:, 0:_lambda_half]
            samples = append(samples, -samples, axis=1)
            
            # projection onto e1
            proj = samples[0, :]
            
            # the largest order statistic
            proj_sorted = sort(proj)
            proj_max[i % n_sample_per_cycle] = proj_sorted[-1]
            
            if (i+1) % n_sample_per_cycle == 0:
                for k, x in enumerate(x_point):
                    ecdf[k] += np.sum(proj_max <= x)
    
    n_worker = 4
    n_sample_per_cycle = 1e3
    n_sample = int(1e4)
    
    _min = -1
    _max = 6
    n_point = 1e4
    x_point = linspace(_min, _max, n_point)
    
    ecdf = [Array('f', zeros(shape(x_point))) for i in range(n_worker)]
    procs = [Process(target=__c_1_lambda_om, args=[_lambda, dim, ecdf[i]]) \
        for i in range(n_worker)]
        
    # Starting the parallel computation
    for p in procs: p.start()
    for p in procs: p.join()
    
    
    ecdf_om = np.mean(ecdf, axis=0) / n_sample
    res = _max - simps(ecdf_om, x_point)
    
    return res


        
def _c_1_lambda_quasi(_lambda, dim):
    """
    Numerical computation for 
    progress coefficient c(1, lambda_om)
    
    """
    from scipy.stats import norm
    import ghalton as gh
    
    seq = gh.Halton(int(dim))

    # for memory concern
    n_sample = 1e3
    n_trial = int(1e1 * n_sample)
    proj_max_quasi = zeros(n_trial)
    
    for i in range(n_trial):
        
        # Halton sequences and quasi-Gaussians
        halton_samples = seq.get(int(_lambda))
        quasi_samples = array([[norm.ppf(halton_samples[k][m]) for m in range(dim)]\
            for k in range(_lambda)]).T
        
        # projection onto e1
        proj_quasi = quasi_samples[0, :]
        
        # the largest order statistic
        proj_sorted_quasi = sort(proj_quasi)
        proj_max_quasi[i] = proj_sorted_quasi[-1]
        
    epdf_quasi = gaussian_kde(proj_max_quasi)
    
    i = lambda x: x * epdf_quasi(x)
    E, err = quad(i, 0, inf)
    
    return E
    
def ___c_1_lambda_quasi(_lambda, dim):
    """
    Numerical computation for 
    progress coefficient c(1, lambda)
    under quasi-random numbers
        
    """
    import ghalton as gh
    
    def __c_1_lambda_quasi(_lambda, dim, proj_max, halton_samples):
       
        from scipy.stats import norm
        
#        seq = gh.Halton(int(dim))
    
        # for memory concern
        n_sample = len(proj_max)
        
        for i in range(n_sample):
            
            # Halton sequences and quasi-Gaussians
                
            quasi_samples = array([[norm.ppf(halton_samples[k][m]) for m in range(dim)]\
                for k in range(_lambda)]).T
            
            # projection onto e1
            proj = quasi_samples[0, :]
            
            # the largest order statistic
            proj_sorted= sort(proj)
            proj_max[i] = proj_sorted[-1]
        
    n_trial = 1e4
    n_worker = 4
    n_sample = int(n_trial / n_worker)
    
    seq = gh.Halton(int(dim))
    halton_samples = seq.get(int(_lambda*n_trial))
    
    proj_max = [Array('f', zeros(n_sample)) for i in range(n_worker)]
    procs = [Process(target=__c_1_lambda_quasi, args=[_lambda, dim, proj_max[i],\
        halton_samples[i*n_sample:(i+1)*n_sample]]) for i in range(n_worker)]
        
    # Starting the parallel computation
    for p in procs: p.start()
    for p in procs: p.join()
    
    all_sample = empty(n_trial)
    i = 0
    for a in proj_max:
        all_sample[i:i+len(a)] = array(a)
        i += len(a)
    
    epdf_om = gaussian_kde(all_sample)
    
    i = lambda x: x * epdf_om(x)
    E, err = quad(i, -inf, inf)
    
    return E
    
def c_1_lambda_quasi(_lambda, dim):
    """
    Numerical computation for 
    progress coefficient c(1, lambda)
    under quasi-random numbers
        
    """
    import ghalton as gh
    
    def __c_1_lambda_quasi(_lambda, dim, ecdf):
       
        from scipy.stats import norm
        
        seq = gh.Halton(int(dim)) 
    
        # for memory concern
        proj_max = zeros(n_sample_per_cycle)
        
        for i in range(n_sample):
            
            # Halton sequences and quasi-Gaussians
            halton_samples = seq.get(int(_lambda))
            quasi_samples = array([[norm.ppf(halton_samples[k][m]) for m in range(dim)]\
                for k in range(_lambda)]).T
            
            # projection onto e1
            proj = quasi_samples[0, :]
            
            # the largest order statistic
            proj_sorted= sort(proj)
            proj_max[i % n_sample_per_cycle] = proj_sorted[-1]
            
            if (i+1) % n_sample_per_cycle == 0:
                for k, x in enumerate(x_point):
                    ecdf[k] += np.sum(proj_max <= x)
        
    n_worker = 4
    n_sample_per_cycle = 1e3
    n_sample = int(1e4)
    
    _min = -1
    _max = 6
    n_point = 1e4
    x_point = linspace(_min, _max, n_point)
    
    ecdf = [Array('f', zeros(n_sample)) for i in range(n_worker)]
    procs = [Process(target=__c_1_lambda_quasi, args=[_lambda, dim, ecdf[i]])\
        for i in range(n_worker)]
        
    # Starting the parallel computation
    for p in procs: p.start()
    for p in procs: p.join()
    
    ecdf_om = np.mean(ecdf, axis=0) / n_sample
    res = _max - simps(ecdf_om, x_point)
    
    return res

if __name__ == '__main__':
    
#    unit_circle(1)
    
#    from scipy.stats import chi
#    a1 = -pi / 6.
#    a2 = - 25*pi / 180.
#    T1 = array([[cos(a1), 0, -sin(a1)],
#                 [0, 1, 0],
#                 [sin(a1), 0, cos(a1)]])
#    
#    T2 = array([[cos(a2), -sin(a2), 0],
#                 [sin(a2), cos(a2), 0],
#                 [0, 0, 1]])
#    
#    v1 = dot(rand_orth_mat(3), eye(3))
#    
#    v2 = dot(T1, dot(T2, v1))
#    v3 = chi.rvs(3, size=3) * v2
#
#    np.disp(v2[1:, 0])
#    np.disp(v2[1:, 1])
#    np.disp(v2[1:, 2])
#    np.disp(v3[1:, 0])
#    np.disp(v3[1:, 1])
#    np.disp(v3[1:, 2])
#    np.disp([norm(v3[:, 0]), norm(v3[:, 1]), norm(v3[:, 2])])
    
    
    X = randn(10, 10)
    A = randn(10, 10)
    C = dot(A, A.T)
    Y = gram_schmidt(X, C)
    pdb.set_trace()
    



#    import hello as h
#    x = randn(5, 3)
#    y1 = h.gram_schmidt(x)
#    y = gram_schmidt(x)
#    y2 = qr(x)[0]
    
#    t = time.clock()
#    for i in range(runs):
#        noise_gen_c(dim)
#    print '%.7f' % ((time.clock() - t) / runs)
#    a = np.array([1,2,3,4])

#    v = np.array([1, 1])
#    A = rand_orth_mat(2)
#    v1 = np.dot(-v, A.T)
#    
#    
#    x1 = np.arange(0, v[0]+.1, .1)
#    x2 = np.append(np.arange(v1[0], 0, .1), 0)
#    y1 = (v[1] / v[0]) * x1
#    y2 = (v1[1] / v1[0]) * x2
#    plt.hold(True)
#    plt.plot(x1, y1, 'b', x2, y2, 'r')
#    
#    delta_x = .01
#    
#    xc = np.append(np.arange(0, 2**.5, delta_x), 2**.5)
#    yc1 = np.power(2 - abs(xc)**2.0, 1.0/2)
#    yc1[-1] = 0
#    yc2 = -yc1
#    plt.plot(xc, yc1, 'm', xc, yc2, 'm', -xc, yc1, 'm', -xc, yc2, 'm')
#    plt.grid(True)
#    plt.show()

#    x = randn(10, 1)
#    y = randn(10, 1)
#    R = rotation_matrix(x, y)
#    print dot(R, x)*norm(y) / norm(x), y
