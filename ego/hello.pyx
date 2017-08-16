import cython
import numpy as np
cimport numpy as np
from numpy cimport uint8_t, double_t, int_t
from cpython cimport bool
from numpy.random import rand, randn
from numpy import arange, empty, arange, pi

AXIS = 0

cdef extern from "math.h":
    double pow(double x, double y)
    double sin(double x)
    double cos(double x)
    double sqrt(double x)
    double exp(double x)
    void* malloc(size_t size)

cdef inline double c_sum1(double* x, int S0, int S1, int N1, int N2):
    cdef int i, j
    cdef double out = 0.0
    for i in range(N1):
        for j in range(N2):
            out += x[i*S0 + j*S1]
    return out
            
cdef inline c_sum2(double* x, double* out, int S0, int S1, int N1, int N2):
    cdef int i, j
    for i in range(N1):
        out[i] = 0.0
        for j in range(N2):
            out[i] += x[i*S0 + j*S1]

cdef inline double c_norm(double* x, int S, int N):
    cdef int i
    cdef double out = 0.0
    if S == 1:
        for i in range(N):
            out += pow(x[i], 2)
    else:
        for i in range(N):
            out += pow(x[i*S], 2)
    return sqrt(out)

cdef inline bint c_any(void* x, int S0, int S1, int N0, int N1, \
    np.uint8_t* out, int t=0):
    cdef int i, j
    cdef bint res = 0
    cdef np.uint8_t* xx1
    cdef int* xx2
    if t == 1:
        xx1 = <uint8_t*>x
        if out == NULL:
            for i in range(N0):
                for j in range(N1):
                    if xx1[i*S0 + j*S1] != 0:
                        res = 1
                        break
                else:
                    continue
                break
        else:
            for i in range(N0):
                out[i] = 0
                for j in range(N1):
                    if xx1[i*S0 + j*S1] != 0:
                        out[i] = 1
                        break
    else:
        xx2 = <int*>x
        if out == NULL:
            for i in range(N0):
                for j in range(N1):
                    if xx2[i*S0 + j*S1] != 0:
                        res = 1
                        break
                else:
                    continue
                break
        else:
            for i in range(N0):
                out[i] = 0
                for j in range(N1):
                    if xx2[i*S0 + j*S1] != 0:
                        out[i] = 1
                        break
    return res

cdef inline void c_outer(double* x, double* y, double* out, Sx, Sy, Nx, Ny):
    cdef int i, j
    for i in range(Nx):
        for j in range(Ny):
            out[i*Ny+j] = x[i*Sx] * y[j*Sy]
            
cdef inline double c_inner(double* x, double* y, Sx, Sy, N):
    cdef int i
    cdef double out = 0.0
    for i in range(N):
        out += x[i*Sx] * y[i*Sy]
    return out
    
cdef inline void c_gram_schmidt(double* x, double* out, S0, S1, N0, N1):
    cdef int i, j, k
    cdef double c, l
    # Copy
    for i in range(N0):
        for k in range(N1):
            out[i+k*N0] = x[i*S0 + k*S1]
        
    # Orthogonormalization
    for i in range(N0):
        for j in range(0, i):
            c = c_inner(&x[i*S0], &out[j], S1, N0, N1)
            for k in range(N1):
                out[i+k*N0] = out[i+k*N0] - out[j+k*N0]*c 
        # Normalization
        l = c_norm(&out[i], N0, N1)
        for k in range(N1):
            out[i+k*N0] = out[i+k*N0] / l
    
cdef void c_sphere(double* x, double* out, int S0, int S1, int N1, int N2):
    cdef int i, j
    # np.sum(np.power())
    for i in range(N1):
        out[i] = 0.0
        for j in range(N2):
            out[i] += pow(x[i*S0 + j*S1], 2)

cdef void c_schwefel(double* x, double* out, int S0, int S1, int N1, int N2):
    cdef int i, j
    # np.cumsum()
    for i in range(N1):
        for j in range(N2-1):
            x[i*S0+(j+1)*S1] = x[i*S0 + (j+1)*S1] + x[i*S0+j*S1]
            
    # np.sum(np.power())
    for i in range(N1):
        out[i] = 0.0
        for j in range(N2):
            out[i] += pow(x[i*S0+j*S1], 2)
    
cdef void c_ackley(double* x, double* out, int S0, int S1, int N1, int N2):
    cdef double c1 = -20, c2 = -0.2,  c3 = 2*pi, tmp
    cdef int i, j
        
    for i in range(N1):
        out[i] = 0.0
        tmp = 0.0
        for j in range(N2):
            out[i] += pow(x[i*S0+j*S1], 2)
            tmp += cos(c3*x[i*S0+j*S1])
        out[i] = c1*exp(c2*sqrt(out[i] / N2)) - exp(tmp / N2) - c1 + exp(1)
        
@cython.boundscheck(False)   
def sum(x, axis=None):
    cdef int nRow, nColumn, S0, S1
    cdef np.ndarray[np.double_t, ndim=1] out, c_x1
    cdef np.ndarray[np.double_t, ndim=2] c_x2
    cdef double out2
    
    if x.ndim == 1:
        c_x1 = x
        nRow = 1
        nColumn = c_x1.shape[0]
        S0 = 0
        S1 = c_x1.strides[0] / sizeof(double)
        out2 = c_sum1(&c_x1[0], S0, S1, nRow, nColumn)
        return out2
    else:
        c_x2 = x
        nRow = c_x2.shape[0]
        nColumn = c_x2.shape[1]
        S0 = c_x2.strides[0] / sizeof(double)
        S1 = c_x2.strides[1] / sizeof(double)
        if axis == None:
            out2 = c_sum1(&c_x2[0, 0], S0, S1, nRow, nColumn) 
            return out2
        elif axis == 1:
            out = empty(nRow)
            c_sum2(&c_x2[0, 0], <double*>out.data, S0, S1, nRow, nColumn) 
        elif axis == 0:
            out = empty(nColumn)
            c_sum2(&c_x2[0, 0], <double*>out.data, S1, S0, nColumn, nRow)
        return out 

@cython.boundscheck(False)   
def norm(x):
    cdef int S, N
    cdef np.ndarray[np.double_t, ndim=1] c_x1
#    cdef np.ndarray[np.double_t, ndim=2] c_x2
    cdef double out
    
    # Vector norm
    if x.ndim == 1:
        c_x1 = x
        N = c_x1.shape[0]
        S = c_x1.strides[0] / sizeof(double)
        out = c_norm(&c_x1[0], S, N)
    else:
        if x.shape[1] == 1:
            c_x1 = x[:, 0]
        else:
            c_x1 = x[0, :]
        N = c_x1.shape[0]
        S = c_x1.strides[0] / sizeof(double)
        out = c_norm(&c_x1[0], S, N) 
    return out

@cython.boundscheck(False)   
def any(x, axis=None):
    cdef int S0, S1, nRow, nColumn
    cdef bint out1
    cdef np.ndarray[np.double_t, ndim=1, cast=True] c_x1
    cdef np.ndarray[np.double_t, ndim=2, cast=True] c_x2
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] c_x1_b
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] c_x2_b
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True, mode='c'] out2
    
    if x.dtype == 'bool':
        if x.ndim == 1:
            if axis == 1:
                raise ValueError(r"'axis' entry is out of bounds")
            c_x1_b = x
            nRow = 1
            nColumn = c_x1_b.shape[0]
            S0 = 0
            S1 = c_x1_b.strides[0] / sizeof(np.uint8_t)
            out1 = c_any(&c_x1_b[0], S0, S1, nRow, nColumn, NULL, 1)
            return out1
        else:
            c_x2_b = x
            nRow = c_x2_b.shape[0]
            nColumn = c_x2_b.shape[1]
            S0 = c_x2_b.strides[0] / sizeof(np.uint8_t)
            S1 = c_x2_b.strides[1] / sizeof(np.uint8_t)
            if axis == 1:
                out2 = empty(nColumn, dtype=np.bool)
                c_any(&c_x2_b[0, 0], S0, S1, nRow, nColumn, <np.uint8_t*>out2.data, 1) 
                return out2
            elif axis == 0:
                out2 = empty(nColumn, dtype=np.bool)
                c_any(&c_x2_b[0, 0], S1, S0, nColumn, nRow, <np.uint8_t*>out2.data, 1)
                return out2
            elif axis == None:
                out1 = c_any(&c_x2_b[0, 0], S0, S1, nRow, nColumn, NULL, 1)
                return out1
    else:
        if x.ndim == 1:
            if axis == 1:
                raise ValueError(r"'axis' entry is out of bounds")
            c_x1 = x
            nRow = 1
            nColumn = c_x1.shape[0]
            S0 = 0
            S1 = c_x1.strides[0] / sizeof(double)
            out1 = c_any(&c_x1[0], S0, S1, nRow, nColumn, NULL)
            return out1
        else:
            c_x2 = x
            nRow = c_x2.shape[0]
            nColumn = c_x2.shape[1]
            S0 = c_x2.strides[0] / sizeof(double)
            S1 = c_x2.strides[1] / sizeof(double)
            if axis == 1:
                out2 = empty(nColumn, dtype=np.bool)
                c_any(&c_x2[0, 0], S0, S1, nRow, nColumn, <np.uint8_t*>out2.data) 
                return out2
            elif axis == 0:
                out2 = empty(nColumn, dtype=np.bool)
                c_any(&c_x2[0, 0], S1, S0, nColumn, nRow, <np.uint8_t*>out2.data)
                return out2
            elif axis == None:
                out1 = c_any(&c_x2[0, 0], S0, S1, nRow, nColumn, NULL)
                return out1

@cython.boundscheck(False)   
def outer(x, y):
    cdef int Nx, Ny, Sx, Sy
    cdef np.ndarray[double_t, ndim=1, cast=True] c_x
    cdef np.ndarray[double_t, ndim=1, cast=True] c_y
    cdef np.ndarray[double_t, ndim=2, mode='c'] out
    
    # adjust x data alignment 
    if x.ndim == 1:
        c_x = x
    elif x.shape[0] == 1:
        c_x = x[0, :]
    elif x.shape[1] == 1:
        c_x = x[:, 0]
    else:
        raise ValueError(r"'x' should be a vector")
        
    # adjust y data alignment
    if y.ndim == 1:
        c_y = y
    elif y.shape[0] == 1:
        c_y = y[0, :]
    elif y.shape[1] == 1:
        c_y = y[:, 0]
    else:
        raise ValueError(r"'y' should be a vector")
    
    Nx = c_x.shape[0]
    Ny = c_y.shape[0]
    Sx = c_x.strides[0] / sizeof(double)
    Sy = c_y.strides[0] / sizeof(double)
    
    out = empty((Nx, Ny), dtype=np.double)
    c_outer(&c_x[0], &c_y[0], &out[0, 0], Sx, Sy, Nx, Ny)
    return out

def gram_schmidt(x):
    cdef int N0, N1, S0, S1
    cdef np.ndarray[double_t, ndim=2, cast=True] c_x
    cdef np.ndarray[double_t, ndim=2, mode='c'] out
    
    if x.ndim ==1 :
        raise ValueError(r"'x' should have two dimensions.")
    N0 = x.shape[0]
    N1 = x.shape[1]
    S0 = x.strides[0] / sizeof(double)
    S1 = x.strides[1] / sizeof(double)
    
    if N0 < N1:
        raise ValueError(r"'x' The columns of x are expected to be linear independent.")
        
    c_x = x
    out = empty((N0, N1))
    c_gram_schmidt(&c_x[0, 0], &out[0, 0], S1, S0, N1, N0)
    return out
    
@cython.boundscheck(False)   
def linear(np.ndarray[np.double_t, ndim=2] x):
    if AXIS == 0:
        return x[0, :]
    else:
        return x[:, 0]
    
@cython.boundscheck(False)   
def sphere(np.ndarray[np.double_t, ndim=2] x):
    cdef int nRow, nColumn, S0, S1
    cdef np.ndarray[np.double_t, ndim=1] out
    
    nRow = x.shape[0]
    nColumn = x.shape[1]
    S0 = x.strides[0] / sizeof(double)
    S1 = x.strides[1] / sizeof(double)
    if AXIS == 1:
        out = empty(nRow)
        c_sphere(&x[0, 0], <double*>out.data, S0, S1, nRow, nColumn)
        
    else:
        out = empty(nColumn)
        c_sphere(&x[0, 0], <double*>out.data, S1, S0, nColumn, nRow)
    return out 

@cython.boundscheck(False)   
def schwefel(np.ndarray[np.double_t, ndim=2] x):
    cdef int nRow, nColumn, S0, S1
    cdef np.ndarray[np.double_t, ndim=1] out
    cdef np.ndarray[np.double_t, ndim=2] xx = x.copy()
    
    nRow = x.shape[0]
    nColumn = x.shape[1]
    S0 = x.strides[0] / sizeof(double)
    S1 = x.strides[1] / sizeof(double)
    if AXIS == 1:
        out = empty(nRow)
        c_schwefel(&xx[0, 0], <double*>out.data, S0, S1, nRow, nColumn)
        
    else:
        out = empty(nColumn)
        c_schwefel(&xx[0, 0], <double*>out.data, S1, S0, nColumn, nRow)
    return out
    
@cython.boundscheck(False) 
def ackley(np.ndarray[np.double_t, ndim=2] x):
    cdef int nRow, nColumn, S0, S1
    cdef np.ndarray[np.double_t, ndim=1] out
    
    nRow = x.shape[0]
    nColumn = x.shape[1]
    S0 = x.strides[0] / sizeof(double)
    S1 = x.strides[1] / sizeof(double)
    if AXIS == 1:
        out = empty(nRow)
        c_ackley(&x[0, 0], <double*>out.data, S0, S1, nRow, nColumn)
        
    else:
        out = empty(nColumn)
        c_ackley(&x[0, 0], <double*>out.data, S1, S0, nColumn, nRow)
    return out  

@cython.boundscheck(False) 
def pairwise_selection(np.ndarray[int_t, ndim=1, cast=True] x, int half, int _mu):
    cdef np.ndarray[int_t, ndim=1, mode='c'] sel
    S = x.strides[0] / sizeof(long)
    N = x.shape[0]
    sel = empty(_mu, dtype='int')
    c_pairwise_selection(<long*>x.data, <long*>sel.data, S, N, half, _mu)
    return sel
    
cdef void c_pairwise_selection(long* x, long* sel, int S, int N, int half, int _mu):
    cdef int* reject = <int*>malloc(_mu*sizeof(int))
    cdef i, k, j = 0
    
    for i in range(_mu):
        reject[i] = -1
    for i in range(N):
        for k in range(j):
            if x[i] == reject[k]:
                break
        else:
            sel[j] = x[i*S]
            if x[i*S] < 2*half:
                if x[i] < half:
                    reject[j] = x[i*S] + half
                else:
                    reject[j] = x[i*S] - half
            j += 1
            if j == _mu:
                break
    
#    
#def axis_parallel_ellipsoid(x):
#    if AXIS == 0:
#        x = x.T
#    if size(x) == shape(x)[0]:
#        N = shape(x)[0]
#    else:
#        N = shape(x)[1]
#    t = x * np.arange(1, N+1)
#    p = power(t, 2)
#    return np.sum(p, axis=1)
#    
#def cigar(x):
#    if AXIS == 0:
#        x = x.T
#    f = x[:, 0]**2 + 1e6*np.sum(x[:, 1::]**2, axis=1)
#    return f
#    
#def tablet(x):
#    if AXIS == 0:
#        x = x.T
#    f = 1e6*x[:, 0]**2 + np.sum(x[:, 1::]**2, axis=1)
#    return f
#    
#def cigatab(x):
#    if AXIS == 0:
#        x = x.T
#    f = x[:, 0]**2 + 1e8*x[:, -1]**2 + 1e4*np.sum(x[:, 1:-1]**2, axis=1)
#    return f
#    
#def ellipsoid(x):
#    if AXIS == 0:
#        x = x.T
#    N = x.shape[1]
#    co = array([1e3**(i/(N-1.0)) for i in arange(0, N)])
#    f = np.sum((co*x)**2, axis=1)
#    return f
#    
#def parabR(x):
#    if AXIS == 0:
#        x = x.T
#    f = -x[:, 0] + 100*np.sum(x[:, 1:]**2, axis=1)
#    return f
#
#def sharpR(x):
#    if AXIS == 0:
#        x = x.T
#    f = -x[:, 0] + 100*sqrt(np.sum(x[:, 1:]**2, axis=1))
#    return f
#    
#def diffpow(x):
#    if AXIS == 0:
#        x = x.T
#    N = x.shape[1]
#    x = np.abs(x)
#    p = 10*arange(0, N) / (N-1.0) + 2
#    f = np.sum(power(x, p), axis=1)
#    return f
#
#def rosenbrock(x):
#    if AXIS == 0:
#        x = x.T
#    sqx = power(x, 2)
#    foobar = power(subtract(x[:, 0:-1], 1), 2)
#    diff = subtract(sqx[:, 0:-1], x[:, 1::])
#    term = 100*power(diff, 2) + foobar
#    return np.sum(term, axis=1)    
#
#def powersum(x):
#    if AXIS == 0:
#        x = x.T
#    if size(x) == shape(x)[0]:
#        N = shape(x)[0]
#    else:
#        N = shape(x)[1]
#    t = np.abs(x)
#    p = power(t, np.arange(2, N+2))
#    return np.sum(p, axis=1)
#    
#def rastrigin(x):
#    if AXIS == 0:
#        x = x.T
#    if size(x) == shape(x)[0]:
#        N = shape(x)[0]
#    else:
#        N = shape(x)[1]
#    f = 10*N + np.sum(power(x, 2)-10*cos(2*pi*x), axis=1)
#    return f
#    
