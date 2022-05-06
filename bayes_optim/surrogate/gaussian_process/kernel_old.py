"""
Created on Wed Aug 23 15:48:28 2017

@author: Hao Wang
@email: wangronin@gmail.com
Note: this module is strongly inspired by the kernel module of the sklearn GaussianProcess kernel implementation.
"""


from abc import abstractmethod
import numpy as np
import math
from scipy.special import kv, gamma

"""
The built-in correlation models submodule for the gaussian_process module.
TODO: The grad of the correlation should be implemented in the 
correlation models

TODO: by default, all the stationary kernels are normalized
"""


class Kernel(object):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, X, Y=None):
        "Evaluate the kernel function"

    def __add__(self, kernel):
        return KernelSum(self, kernel)

    def __mul__(self, kernel):
        return KernelProduct(self, kernel)

    def __rmul__(self, kernel):
        return KernelProduct(self, kernel)


class ConstantKernel(Kernel):
    def __init__(self, sigma2=1.0):
        self.sigma2 = sigma2

    def __call__(self, X, Y=None):
        pass


class CompositeKernel(Kernel):
    """
    The space of kernels is closed under addition and multiplication
    """

    def __init__(self):
        pass


class KernelSum(CompositeKernel):
    def __call__(self, X, Y=None):
        return self.K1(X, Y) + self.K2(X, Y)


class KernelProduct(CompositeKernel):
    def __call__(self, X, Y=None):
        return self.K1(X, Y) * self.K2(X, Y)


class HammingKernel(Kernel):
    """
    Kernel function for categorical variables using Hamming distance
    """

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X


class StationaryKernel(Kernel):
    pass


class Matern(StationaryKernel):
    def __init__(self, theta=None, bounds=None, nu=1.5):
        self.nu = nu
        self.theta = theta
        self.bounds = self.theta_bounds() if bounds else bounds

    def __call__(self, X):

        theta = np.asarray(theta, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        if X.ndim > 1:
            n_features = X.shape[1]
        else:
            n_features = 1

        if theta.size == 1:
            dists = np.sqrt(theta[0] * np.sum(X ** 2, axis=1))
        else:
            dists = np.sqrt(np.sum(theta.reshape(1, n_features) * X ** 2, axis=1))
        # Matern 1/2
        if nu == 0.5:
            K = np.exp(-dists)
        # Matern 3/2
        elif nu == 1.5:

            K = dists * math.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        # Matern 5/2
        elif nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K ** 2 / 3.0) * np.exp(-K)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * nu) * K
            K.fill((2 ** (1.0 - nu)) / gamma(nu))
            K *= tmp ** nu
            K *= kv(nu, tmp)

    def dx(self, X):
        pass

    def dtheta(self, X):
        c = np.sqrt(3)
        D = np.sqrt(np.sum(theta * diff, axis=-1))

        if nu == 0.5:
            grad = -diff * theta / D * R
        elif nu == 1.5:
            grad = -3 * np.exp(-c * D)[..., np.newaxis] * diff / 2.0
        elif nu == 2.5:
            pass


# Note: the c-wrapper for kernel functions
# def matern(theta, X, eval_Dx=False, eval_Dtheta=False,
#           length_scale_bounds=(1e-5, 1e5), nu=1.5):
#    n_max = dim*(dim-1)/2
#
#    index = (rand(n_max) < pm) * arange(1, n_max+1)
#    index = index[index > 0]
#    index = index - 1
#
#    n_rotation = index.shape[0]
#    rotations = angle * (2*rand(n_rotation) - 1)
##        res = zeros((dim, dim))
#
#    # Prepare the arguments for C function
#    POS = index.ctypes.data_as(POINTER(c_uint))
#    ROTATIONS = rotations.ctypes.data_as(POINTER(c_double))
#    RES = operator.ctypes.data_as(POINTER(c_double))
#
#    # Invoke the C code for dense matrix multiplication
#    c_noisy_gen(dim, POS, n_rotation, ROTATIONS, RES)


def matern(theta, X, eval_Dx=False, eval_Dtheta=False, length_scale_bounds=(1e-5, 1e5), nu=1.5):
    """
    theta = np.asarray(theta, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    if d.ndim > 1:
        n_features = d.shape[1]
    else:
        n_features = 1

    if theta.size == 1:
        return np.exp(-theta[0] * np.sum(d ** 2, axis=1))
    elif theta.size != n_features:
        raise ValueError("Length of theta must be 1 or %s" % n_features)
    else:
        return np.exp(-np.sum(theta.reshape(1, n_features) * d ** 2, axis=1))

    """
    theta = np.asarray(theta, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim > 1:
        n_features = X.shape[1]
    else:
        n_features = 1

    if theta.size == 1:
        dists = np.sqrt(theta[0] * np.sum(X ** 2, axis=1))
    else:
        dists = np.sqrt(np.sum(theta.reshape(1, n_features) * X ** 2, axis=1))

    # Matern 1/2
    if nu == 0.5:
        K = np.exp(-dists)
    # Matern 3/2
    elif nu == 1.5:

        K = dists * math.sqrt(3)
        K = (1.0 + K) * np.exp(-K)
    # Matern 5/2
    elif nu == 2.5:
        K = dists * math.sqrt(5)
        K = (1.0 + K + K ** 2 / 3.0) * np.exp(-K)
    else:  # general case; expensive to evaluate
        K = dists
        K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
        tmp = math.sqrt(2 * nu) * K
        K.fill((2 ** (1.0 - nu)) / gamma(nu))
        K *= tmp ** nu
        K *= kv(nu, tmp)

    if eval_Dx:
        pass

    # infoert from upper-triangular matrix to square matrix
    # K = squareform(K)
    # np.fill_diagonal(K, 1)
    if eval_Dtheta:
        pass
    #        # We need to recompute the pairwise dimension-wise distances
    #        if self.anisotropic:
    #            D = (X[:, np.newaxis, :] - X[np.newaxis, :, :])**2 \
    #                / (length_scale ** 2)
    #        else:
    #            D = squareform(dists**2)[:, :, np.newaxis]
    #
    #        if self.nu == 0.5:
    #            K_gradient = K[..., np.newaxis] * D \
    #                / np.sqrt(D.sum(2))[:, :, np.newaxis]
    #            K_gradient[~np.isfinite(K_gradient)] = 0
    #        elif self.nu == 1.5:
    #            K_gradient = \
    #                3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
    #        elif self.nu == 2.5:
    #            tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
    #            K_gradient = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)
    #        else:
    #            # approximate gradient numerically
    #            def f(theta):  # helper function
    #                return self.clone_with_theta(theta)(X, Y)
    #            return K, _approx_fprime(self.theta, f, 1e-10)
    #
    #        if not self.anisotropic:
    #            return K, K_gradient[:, :].sum(-1)[:, :, np.newaxis]
    #        else:
    #            return K, K_gradient
    return K


def absolute_exponential(theta, d):
    """
    Absolute exponential autocorrelation model.
    (Ornstein-Uhlenbeck stochastic process)::

                                          n
        theta, d --> r(theta, d) = exp(  sum  - theta_i * |d_i| )
                                        i = 1

    Parameters
    ----------
    theta : array_like
        An array with shape 1 (isotropic) or n (anisotropic) giving the
        autocorrelation parameter(s).

    d : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.

    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) containing the values of the
        autocorrelation model.
    """
    theta = np.asarray(theta, dtype=np.float64)
    d = np.abs(np.asarray(d, dtype=np.float64))

    if d.ndim > 1:
        n_features = d.shape[1]
    else:
        n_features = 1

    if theta.size == 1:
        return np.exp(-theta[0] * np.sum(d, axis=1))
    elif theta.size != n_features:
        raise ValueError("Length of theta must be 1 or %s" % n_features)
    else:
        return np.exp(-np.sum(theta.reshape(1, n_features) * d, axis=1))


def squared_exponential(theta, d):
    """
    Squared exponential correlation model (Radial Basis Function).
    (Infinitely differentiable stochastic process, very smooth)::

                                          n
        theta, d --> r(theta, d) = exp(  sum  - theta_i * (d_i)^2 )
                                        i = 1

    Parameters
    ----------
    theta : array_like
        An array with shape 1 (isotropic) or n (anisotropic) giving the
        autocorrelation parameter(s).

    d : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.

    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) containing the values of the
        autocorrelation model.
    """

    theta = np.asarray(theta, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    if d.ndim > 1:
        n_features = d.shape[1]
    else:
        n_features = 1

    if theta.size == 1:
        return np.exp(-theta[0] * np.sum(d ** 2, axis=1))
    elif theta.size != n_features:
        raise ValueError("Length of theta must be 1 or %s" % n_features)
    else:
        return np.exp(-np.sum(theta.reshape(1, n_features) * d ** 2, axis=1))


def generalized_exponential(theta, d):
    """
    Generalized exponential correlation model.
    (Useful when one does not know the smoothness of the function to be
    predicted.)::

                                          n
        theta, d --> r(theta, d) = exp(  sum  - theta_i * |d_i|^p )
                                        i = 1

    Parameters
    ----------
    theta : array_like
        An array with shape 1+1 (isotropic) or n+1 (anisotropic) giving the
        autocorrelation parameter(s) (theta, p).

    d : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.

    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) with the values of the autocorrelation
        model.
    """

    theta = np.asarray(theta, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    if d.ndim > 1:
        n_features = d.shape[1]
    else:
        n_features = 1

    lth = theta.size
    if n_features > 1 and lth == 2:
        theta = np.hstack([np.repeat(theta[0], n_features), theta[1]])
    elif lth != n_features + 1:
        raise Exception("Length of theta must be 2 or %s" % (n_features + 1))
    else:
        theta = theta.reshape(1, lth)

    td = theta[:, 0:-1].reshape(1, n_features) * np.abs(d) ** theta[:, -1]
    r = np.exp(-np.sum(td, 1))

    return r


def pure_nugget(theta, d):
    """
    Spatial independence correlation model (pure nugget).
    (Useful when one wants to solve an ordinary least squares problem!)::

                                           n
        theta, d --> r(theta, d) = 1 if   sum |d_i| == 0
                                         i = 1
                                   0 otherwise

    Parameters
    ----------
    theta : array_like
        None.

    d : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.

    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) with the values of the autocorrelation
        model.
    """

    theta = np.asarray(theta, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    n_eval = d.shape[0]
    r = np.zeros(n_eval)
    r[np.all(d == 0.0, axis=1)] = 1.0

    return r


def cubic(theta, d):
    """
    Cubic correlation model::

        theta, d --> r(theta, d) =
          n
         prod max(0, 1 - 3(theta_j*d_ij)^2 + 2(theta_j*d_ij)^3) ,  i = 1,...,m
        j = 1

    Parameters
    ----------
    theta : array_like
        An array with shape 1 (isotropic) or n (anisotropic) giving the
        autocorrelation parameter(s).

    d : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.

    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) with the values of the autocorrelation
        model.
    """

    theta = np.asarray(theta, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    if d.ndim > 1:
        n_features = d.shape[1]
    else:
        n_features = 1

    lth = theta.size
    if lth == 1:
        td = np.abs(d) * theta
    elif lth != n_features:
        raise Exception("Length of theta must be 1 or " + str(n_features))
    else:
        td = np.abs(d) * theta.reshape(1, n_features)

    td[td > 1.0] = 1.0
    ss = 1.0 - td ** 2.0 * (3.0 - 2.0 * td)
    r = np.prod(ss, 1)

    return r
