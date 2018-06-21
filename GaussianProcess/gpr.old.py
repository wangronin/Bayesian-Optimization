# -*- coding: utf-8 -*-

# Author: Hao Wang <wangronin@gmail.com>
#         Bas van Stein <bas9112@gmail.com>


from __future__ import print_function

import pdb
import numpy as np
from numpy import log, pi, log10

from scipy import linalg, optimize
from scipy.linalg import cho_solve
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import kv, gamma

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.utils import check_random_state, check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

import math
import warnings


"""
The built-in regression models submodule for the gaussian_process module.
"""

def constant(x):
    """
    Zero order polynomial (constant, p = 1) regression model.

    x --> f(x) = 1

    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    """
    x = np.asarray(x, dtype=np.float64)
    n_eval = x.shape[0]
    f = np.ones([n_eval, 1])
    return f


def linear(x):
    """
    First order polynomial (linear, p = n+1) regression model.

    x --> f(x) = [ 1, x_1, ..., x_n ].T

    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    """
    x = np.asarray(x, dtype=np.float64)
    n_eval = x.shape[0]
    f = np.hstack([np.ones([n_eval, 1]), x])
    return f


def quadratic(x):
    """
    Second order polynomial (quadratic, p = n*(n-1)/2+n+1) regression model.

    x --> f(x) = [ 1, { x_i, i = 1,...,n }, { x_i * x_j,  (i,j) = 1,...,n } ].T
                                                          i > j

    Parameters
    ----------
    x : array_like
        An array with shape (n_eval, n_features) giving the locations x at
        which the regression model should be evaluated.

    Returns
    -------
    f : array_like
        An array with shape (n_eval, p) with the values of the regression
        model.
    """

    x = np.asarray(x, dtype=np.float64)
    n_eval, n_features = x.shape
    f = np.hstack([np.ones([n_eval, 1]), x])
    for k in range(n_features):
        f = np.hstack([f, x[:, k, np.newaxis] * x[:, k:]])

    return f


"""
The built-in correlation models submodule for the gaussian_process module.
TODO: The grad of the correlation should be implemented in the correlation models
"""

def matern(theta, X, eval_Dx=False, eval_Dtheta=False,
           length_scale_bounds=(1e-5, 1e5), nu=1.5):
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
        K = (1. + K) * np.exp(-K)
    # Matern 5/2
    elif nu == 2.5:
        K = dists * math.sqrt(5)
        K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
    else:  # general case; expensive to evaluate
        K = dists
        K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
        tmp = (math.sqrt(2 * nu) * K)
        K.fill((2 ** (1. - nu)) / gamma(nu))
        K *= tmp ** nu
        K *= kv(nu, tmp)


    if eval_Dx:
        pass

    # convert from upper-triangular matrix to square matrix
    #K = squareform(K)
    #np.fill_diagonal(K, 1)

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
        return np.exp(- theta[0] * np.sum(d, axis=1))
    elif theta.size != n_features:
        raise ValueError("Length of theta must be 1 or %s" % n_features)
    else:
        return np.exp(- np.sum(theta.reshape(1, n_features) * d, axis=1))


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
    r = np.exp(- np.sum(td, 1))

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
    r[np.all(d == 0., axis=1)] = 1.

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

    td[td > 1.] = 1.
    ss = 1. - td ** 2. * (3. - 2. * td)
    r = np.prod(ss, 1)

    return r



MACHINE_EPSILON = np.finfo(np.double).eps


def l1_cross_distances(X):
    """
    Computes the nonzero componentwise L1 cross-distances between the vectors
    in X.

    Parameters
    ----------

    X: array_like
        An array with shape (n_samples, n_features)

    Returns
    -------

    D: array with shape (n_samples * (n_samples - 1) / 2, n_features)
        The array of componentwise L1 cross-distances.

    ij: arrays with shape (n_samples * (n_samples - 1) / 2, 2)
        The indices i and j of the vectors in X associated to the cross-
        distances in D: D[k] = np.abs(X[ij[k, 0]] - Y[ij[k, 1]]).
    """
    X = check_array(X)
    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0
    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1):n_samples])

    return D, ij


# TODO: remove the dependences from sklearn
class GaussianProcess(BaseEstimator, RegressorMixin):
    """The Gaussian Process model class.

    Read more in the :ref:`User Guide <gaussian_process>`.

    Parameters
    ----------
    regr : string or callable, optional
        A regression function returning an array of outputs of the linear
        regression functional basis. The number of observations n_samples
        should be greater than the size p of this basis.
        Default assumes a simple constant regression trend.
        Available built-in regression models are::

            'constant', 'linear', 'quadratic'

    corr : string or callable, optional
        A stationary autocorrelation function returning the autocorrelation
        between two points x and x'.
        Default assumes a squared-exponential autocorrelation model.
        Built-in correlation models are::

            'absolute_exponential', 'squared_exponential',
            'generalized_exponential', 'cubic', 'linear', 'matern'

    beta0 : double array_like, optional
        The regression weight vector to perform Ordinary Kriging (OK).
        Default assumes Universal Kriging (UK) so that the vector beta of
        regression weights is estimated using the maximum likelihood
        principle.

    storage_mode : string, optional
        A string specifying whether the Cholesky decomposition of the
        correlation matrix should be stored in the class (storage_mode =
        'full') or not (storage_mode = 'light').
        Default assumes storage_mode = 'full', so that the
        Cholesky decomposition of the correlation matrix is stored.
        This might be a useful parameter when one is not interested in the
        MSE and only plan to estimate the BLUP, for which the correlation
        matrix is not required.

    verbose : boolean, optional
        A boolean specifying the verbose level.
        Default is verbose = False.

    theta0 : double array_like, optional
        An array with shape (n_features, ) or (1, ).
        The parameters in the autocorrelation model.
        If thetaL and thetaU are also specified, theta0 is considered as
        the starting point for the maximum likelihood estimation of the
        best set of parameters.
        Default assumes isotropic autocorrelation model with theta0 = 1e-1.

    thetaL : double array_like, optional
        An array with shape matching theta0's.
        Lower bound on the autocorrelation parameters for maximum
        likelihood estimation.
        Default is None, so that it skips maximum likelihood estimation and
        it uses theta0.

    thetaU : double array_like, optional
        An array with shape matching theta0's.
        Upper bound on the autocorrelation parameters for maximum
        likelihood estimation.
        Default is None, so that it skips maximum likelihood estimation and
        it uses theta0.

    normalize : boolean, optional
        Input X and observations y are centered and reduced wrt
        means and standard deviations estimated from the n_samples
        observations provided.
        Default is normalize = True so that data is normalized to ease
        maximum likelihood estimation.

    TODO: the nugget behaves differently than what is described here...
    nugget : double or ndarray, optional
        Introduce a nugget effect to allow smooth predictions from noisy
        data.  If nugget is an ndarray, it must be the same length as the
        number of data points used for the fit.
        The nugget is added to the diagonal of the assumed training covariance;
        in this way it acts as a Tikhonov regularization in the problem.  In
        the special case of the squared exponential correlation function, the
        nugget mathematically represents the variance of the input values.
        Default assumes a nugget close to machine precision for the sake of
        robustness (nugget = 10. * MACHINE_EPSILON).

    optimizer : string, optional
        A string specifying the optimization algorithm to be used.
        Default uses 'fmin_cobyla' algorithm from scipy.optimize.
        Available optimizers are::

            'fmin_cobyla', 'Welch'

        'Welch' optimizer is dued to Welch et al., see reference [WBSWM1992]_.
        It consists in iterating over several one-dimensional optimizations
        instead of running one single multi-dimensional optimization.

    random_start : int, optional
        The number of times the Maximum Likelihood Estimation should be
        performed from a random starting point.
        The first MLE always uses the specified starting point (theta0),
        the next starting points are picked at random according to an
        exponential distribution (log-uniform on [thetaL, thetaU]).
        Default does not use random starting point (random_start = 1).

    random_state: integer or numpy.RandomState, optional
        The generator used to shuffle the sequence of coordinates of theta in
        the Welch optimizer. If an integer is given, it fixes the seed.
        Defaults to the global numpy random number generator.


    Attributes
    ----------
    theta_ : array
        Specified theta OR the best set of autocorrelation parameters (the \
        sought maximizer of the reduced likelihood function).

    reduced_likelihood_function_value_ : array
        The optimal reduced likelihood function value.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.gaussian_process import GaussianProcess
    >>> X = np.array([[1., 3., 5., 6., 7., 8.]]).T
    >>> y = (X * np.sin(X)).ravel()
    >>> gp = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1.)
    >>> gp.fit(X, y)                                      # doctest: +ELLIPSIS
    GaussianProcess(beta0=None...
            ...

    Notes
    -----
    The presentation implementation is based on a translation of the DACE
    Matlab toolbox, see reference [NLNS2002]_.

    References
    ----------

    .. [NLNS2002] `H.B. Nielsen, S.N. Lophaven, H. B. Nielsen and J.
        Sondergaard.  DACE - A MATLAB Kriging Toolbox.` (2002)
        http://imedea.uib-csic.es/master/cambioglobal/Modulo_V_cod101615/Lab/lab_maps/krigging/DACE-krigingsoft/dace/dace.pdf

    .. [WBSWM1992] `W.J. Welch, R.J. Buck, J. Sacks, H.P. Wynn, T.J. Mitchell,
        and M.D.  Morris (1992). Screening, predicting, and computer
        experiments.  Technometrics, 34(1) 15--25.`
        http://www.jstor.org/stable/1269548
    """

    _regression_types = {
        'constant': constant,
        'linear': linear,
        'quadratic': quadratic}

    _correlation_types = {
        'absolute_exponential': absolute_exponential,
        'squared_exponential': squared_exponential,
        'generalized_exponential': generalized_exponential,
        'cubic': cubic,
        'matern': matern,
        'linear': linear}

    _optimizer_types = [
        'BFGS',
        'fmin_cobyla'
        ]

    # 10. * MACHINE_EPSILON
    def __init__(self, regr='constant', corr='squared_exponential', beta0=None,
                 verbose=False, theta0=1e-1, thetaL=None, thetaU=None,
                 optimizer='BFGS', random_start=1, normalize=False,
                 nugget=None, nugget_estim=False, x_lb=None, x_ub=None,
                 random_state=None):

        self.regr = regr
        self.corr = corr
        self.beta0 = beta0
        self.verbose = verbose
        self.theta0 = theta0
        self.thetaL = thetaL
        self.thetaU = thetaU
        self.normalize = normalize
        self.optimizer = optimizer
        self.random_start = random_start
        self.random_state = random_state
        self.wait_iter = int(random_start)

        self.noise_var = np.atleast_1d(nugget) if nugget is not None else None
        self.noisy = True if (self.noise_var is not None) or self.nugget_estim else False
        self.nugget_estim = True if nugget_estim else False

        # three cases to compute the log-likelihood function
        if not self.noisy:
            self.log_likelihood_mode = 'noiseless'
        elif self.nugget_estim:
            self.log_likelihood_mode = 'nugget_estim'
        else:
            self.log_likelihood_mode = 'noisy'

    def fit(self, X, y):
        """
        The Gaussian Process model fitting method.

        Parameters
        ----------
        X : double array_like
            An array with shape (n_samples, n_features) with the input at which
            observations were made.

        y : double array_like
            An array with shape (n_samples, ) or shape (n_samples, n_targets)
            with the observations of the output to be predicted.

        Returns
        -------
        gp : self
            A fitted Gaussian Process model object awaiting data to perform
            predictions.
        """
        # Run input checks

        self._check_params()

        self.random_state = check_random_state(self.random_state)

        # Force data to 2D numpy.array
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self.y_ndim_ = y.ndim
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # Check shapes of DOE & observations
        n_samples, n_features = X.shape
        _, n_targets = y.shape

        # Run input checks
        self._check_params(n_samples)

        # Normalize data or don't
        if self.normalize:
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            y_mean = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            X_std[X_std == 0.] = 1.
            y_std[y_std == 0.] = 1.
            # center and scale X if necessary
            X = (X - X_mean) / X_std
            y = (y - y_mean) / y_std
        else:
            X_mean = np.zeros(1)
            X_std = np.ones(1)
            y_mean = np.zeros(1)
            y_std = np.ones(1)

        # Calculate matrix of distances D between samples
        D, ij = l1_cross_distances(X)
        if (np.min(np.sum(D, axis=1)) == 0.
                and self.corr != pure_nugget):
            raise Exception("Multiple input features cannot have the same"
                            " target value.")

        # Regression matrix and parameters
        F = self.regr(X)
        n_samples_F = F.shape[0]
        if F.ndim > 1:
            p = F.shape[1]
        else:
            p = 1
        if n_samples_F != n_samples:
            raise Exception("Number of rows in F and X do not match. Most "
                            "likely something is going wrong with the "
                            "regression model.")
        if p > n_samples_F:
            raise Exception(("Ordinary least squares problem is undetermined "
                             "n_samples=%d must be greater than the "
                             "regression model size p=%d.") % (n_samples, p))
        if self.beta0 is not None:
            if self.beta0.shape[0] != p:
                raise Exception("Shapes of beta0 and F do not match.")

        # Set attributes
        self.X = X
        self.y = y
        self.D = D
        self.ij = ij
        self.F = F
        self.X_mean, self.X_std = X_mean, X_std
        self.y_mean, self.y_std = y_mean, y_std

        # Determine Gaussian Process model parameters
        if self.thetaL is not None and self.thetaU is not None:
            # Maximum Likelihood Estimation of the parameters
            if self.verbose:
                print("Performing Maximum Likelihood Estimation of the "
                      "autocorrelation parameters...")
            self.theta_, self.reduced_likelihood_function_value_, par = \
                self._arg_max_reduced_likelihood_function()
            if np.isinf(self.reduced_likelihood_function_value_):
                raise Exception("Bad parameter region. "
                                "Try increasing upper bound")

        else:
            # Given parameters
            if self.verbose:
                print("Given autocorrelation parameters. "
                      "Computing Gaussian Process model parameters...")
            par = {}
            self.theta_ = self.theta0
            self.reduced_likelihood_function_value_, par = \
                self.log_likelihood_function(self.theta_, par)
            if np.isinf(self.reduced_likelihood_function_value_):
                raise Exception("Bad point. Try increasing theta0.")

        self.noise_var = par['noise_var']
        self.sigma2 = par['sigma2']
        self.rho = par['rho']
        self.Yt = par['Yt']
        self.C = par['C']
        self.Ft = par['Ft']
        self.G = par['G']
        self.Q = par['Q']

        # compute for beta and gamma
        self.compute_beta_gamma()

        return self

    def predict(self, X, eval_MSE=False, batch_size=None):
        """
        This function evaluates the Gaussian Process model at x.

        Parameters
        ----------
        X : array_like
            An array with shape (n_eval, n_features) giving the point(s) at
            which the prediction(s) should be made.

        eval_MSE : boolean, optional
            A boolean specifying whether the Mean Squared Error should be
            evaluated or not.
            Default assumes evalMSE = False and evaluates only the BLUP (mean
            prediction).

        batch_size : integer, optional
            An integer giving the maximum number of points that can be
            evaluated simultaneously (depending on the available memory).
            Default is None so that all given points are evaluated at the same
            time.

        Returns
        -------
        y : array_like, shape (n_samples, ) or (n_samples, n_targets)
            An array with shape (n_eval, ) if the Gaussian Process was trained
            on an array of shape (n_samples, ) or an array with shape
            (n_eval, n_targets) if the Gaussian Process was trained on an array
            of shape (n_samples, n_targets) with the Best Linear Unbiased
            Prediction at x.

        MSE : array_like, optional (if eval_MSE == True)
            An array with shape (n_eval, ) or (n_eval, n_targets) as with y,
            with the Mean Squared Error at x.
        """
        check_is_fitted(self, "X")

        # Check input shapes
        X = check_array(X)
        n_eval, _ = X.shape
        n_samples, n_features = self.X.shape
        n_samples_y, n_targets = self.y.shape

        # Run input checks
        self._check_params(n_samples)

        if X.shape[1] != n_features:
            raise ValueError(("The number of features in X (X.shape[1] = %d) "
                              "should match the number of features used "
                              "for fit() "
                              "which is %d.") % (X.shape[1], n_features))

        if batch_size is None:
            # No memory management
            # (evaluates all given points in a single batch run)

            # Normalize input
            X = (X - self.X_mean) / self.X_std

            # Initialize output
            y = np.zeros(n_eval)
            if eval_MSE:
                MSE = np.zeros(n_eval)

            # Get pairwise componentwise L1-distances to the input training set
            dx = manhattan_distances(X, Y=self.X, sum_over_features=False)
            # Get regression function and correlation
            f = self.regr(X)
            r = self.corr(self.theta_, dx).reshape(n_eval, n_samples)

            # Scaled predictor
            y_ = np.dot(f, self.beta) + np.dot(r, self.gamma)

            # Predictor
            y = (self.y_mean + self.y_std * y_).reshape(n_eval, n_targets)

            if self.y_ndim_ == 1:
                y = y.ravel()

            # Mean Squared Error
            if eval_MSE:
                rt = linalg.solve_triangular(self.C, r.T, lower=True)

                if self.beta0 is None:
                    # Universal Kriging
                    u = linalg.solve_triangular(self.G.T,
                                                np.dot(self.Ft.T, rt) - f.T,
                                                lower=True)
                else:
                    # Ordinary Kriging
                    u = np.zeros((n_targets, n_eval))

                MSE = np.dot(self.sigma2.reshape(n_targets, 1),
                             (1. - (rt ** 2.).sum(axis=0)
                              + (u ** 2.).sum(axis=0))[np.newaxis, :])
                MSE = np.sqrt((MSE ** 2.).sum(axis=0) / n_targets)

                # Mean Squared Error might be slightly negative depending on
                # machine precision: force to zero!
                MSE[MSE < 0.] = 0.

                if self.y_ndim_ == 1:
                    MSE = MSE.ravel()

                return y, MSE

            else:

                return y

        else:
            # Memory management

            if type(batch_size) is not int or batch_size <= 0:
                raise Exception("batch_size must be a positive integer")

            if eval_MSE:

                y, MSE = np.zeros(n_eval), np.zeros(n_eval)
                for k in range(max(1, n_eval / batch_size)):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size + 1, n_eval + 1])
                    y[batch_from:batch_to], MSE[batch_from:batch_to] = \
                        self.predict(X[batch_from:batch_to],
                                     eval_MSE=eval_MSE, batch_size=None)

                return y, MSE

            else:

                y = np.zeros(n_eval)
                for k in range(max(1, n_eval / batch_size)):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size + 1, n_eval + 1])
                    y[batch_from:batch_to] = \
                        self.predict(X[batch_from:batch_to],
                                     eval_MSE=eval_MSE, batch_size=None)

                return y


    def corr_grad_theta(self, theta, X, R, nu=1.5):
        # Check input shapes
        X = np.atleast_2d(X)
        n_eval, _ = X.shape
        n_features = self.X.shape[1]

        if _ != n_features:
            raise Exception('x does not have the right size!')

        diff = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2.

        if self.corr_type == 'squared_exponential':
            grad = -diff * R[..., np.newaxis]

        elif self.corr_type == 'matern':
            c = np.sqrt(3)
            D = np.sqrt(np.sum(theta * diff, axis=-1))

            if nu == 0.5:
                grad = - diff * theta / D * R
            elif nu == 1.5:
                grad = -3 * np.exp(-c * D)[..., np.newaxis] * diff / 2.
            elif nu == 2.5:
                pass

        elif self.corr_type == 'absolute_exponential':
            grad = -np.sqrt(diff) * R[..., np.newaxis]
        elif self.corr_type == 'generalized_exponential':
            pass
        elif self.corr_type == 'cubic':
            pass
        elif self.corr_type == 'linear':
            pass

        return grad

    def correlation_matrix(self, theta, X=None):
        D = self.D
        ij = self.ij
        n_samples = self.X.shape[0]

        # Set up R
        r = self.corr(theta, D)

        R = np.eye(n_samples)
        R[ij[:, 0], ij[:, 1]] = r
        R[ij[:, 1], ij[:, 0]] = r

        return R

    def _compute_aux_var(self, R):
        # Cholesky decomposition of R: Note that this matrix R can be singular
        # change notation from 'C' to 'L': 'L'ower triagular component...
        try:
            L = linalg.cholesky(R, lower=True)
        except linalg.LinAlgError as e:
            raise e

        # Get generalized least squares solution
        Ft = linalg.solve_triangular(L, self.F, lower=True)
        Yt = linalg.solve_triangular(L, self.y, lower=True)

        # compute rho
        Q, G = linalg.qr(Ft, mode='economic')
        rho = Yt - np.dot(Q.dot(Q.T), Yt)

        return L, Ft, Yt, Q, G, rho

    def log_likelihood_function(self, hyper_par, par_out=None, eval_grad=False):
        """
        TODO: rewrite the documentation here
        TODO: maybe eval_hessian in the future?...
        This function determines the BLUP parameters and evaluates the reduced
        likelihood function for the given autocorrelation parameters theta.

        Maximizing this function wrt the autocorrelation parameters theta is
        equivalent to maximizing the likelihood of the assumed joint Gaussian
        distribution of the observations y evaluated onto the design of
        experiments X.

        Parameters
        ----------
        theta : array_like, optional
            An array containing the autocorrelation parameters at which the
            Gaussian Process model parameters should be determined.
            Default uses the built-in autocorrelation parameters
            (ie ``theta = self.theta_``).

        Returns
        -------
        log_likelihood_function_value : double
            The value of the reduced likelihood function associated to the
            given autocorrelation parameters theta.

        par : dict
            A dictionary containing the requested Gaussian Process model
            parameters:

                sigma2
                        Gaussian Process variance.
                beta
                        Generalized least-squares regression weights for
                        Universal Kriging or given beta0 for Ordinary
                        Kriging.
                gamma
                        Gaussian Process weights.
                C
                        Cholesky decomposition of the correlation matrix [R].
                Ft
                        Solution of the linear equation system : [R] x Ft = F
                G
                        QR decomposition of the matrix Ft.
        """
        check_is_fitted(self, "X")

        # Log-likelihood
        log_likelihood = -np.inf

        # Retrieve data
        n_samples, n_features = self.X.shape
        n_hyper_par = len(hyper_par)

        if self.log_likelihood_mode == 'noiseless':
            theta = hyper_par
            noise_var = 0

            R = self.correlation_matrix(theta)
            L, Ft, Yt, Q, G, rho = self._compute_aux_var(R)
            sigma2 = (rho ** 2.).sum(axis=0) / n_samples

            log_likelihood = -0.5 * (n_samples * log(2.*pi*sigma2) \
                + 2. * np.log(np.diag(L)).sum() + n_samples)

        elif self.log_likelihood_mode == 'nugget_estim':
            theta, alpha = hyper_par[:-1], hyper_par[-1]

            R0 = self.correlation_matrix(theta)
            R = alpha * R0 + (1 - alpha) * np.eye(n_samples)

            try:
                L, Ft, Yt, Q, G, rho = self._compute_aux_var(R)
            except linalg.LinAlgError:
                return (log_likelihood, np.zeros(n_hyper_par, 1)) if eval_grad else log_likelihood

            sigma2_total = (rho ** 2.).sum(axis=0) / n_samples
            sigma2, noise_var = alpha * sigma2_total, (1-alpha) * sigma2_total

            log_likelihood = -0.5 * (n_samples * log(2.*pi*sigma2_total) \
                + 2. * np.log(np.diag(L)).sum() + n_samples)

        elif self.log_likelihood_mode == 'noisy':
            theta, sigma2 = hyper_par[:-1], hyper_par[-1]
            noise_var = self.noise_var
            sigma2_total = sigma2 + noise_var
            sd_total = np.sqrt(sigma2_total)

            R0 = self.correlation_matrix(theta)
            C = sigma2 * R0 + noise_var * np.eye(n_samples)
            R = C / sigma2_total
            L, Ft, Yt, Q, G, rho = self._compute_aux_var(C)

            log_likelihood = -0.5 * (n_samples * log(2.*pi) \
                + 2. * np.log(np.diag(L)).sum() + np.dot(rho.T, rho))

            L = L / sigma2_total
            Ft, Yt, rho = map(lambda x: x*sigma2_total, [Ft, Yt, rho])
            Q, R = Q * sd_total, R * sd_total

        sigma2 *= self.y_std ** 2.

        if par_out is not None:
            par_out['sigma2'] = sigma2
            par_out['noise_var'] = noise_var
            par_out['rho'] = rho
            par_out['Yt'] = Yt
            # TODO: change variable 'C' --> 'L'
            par_out['C'] = L
            par_out['Ft'] = Ft
            par_out['G'] = G
            par_out['Q'] = Q

        # for verification
        # TODO: remove this in the future
        if np.exp(log_likelihood) > 1:
            return -np.inf, np.zeros((n_hyper_par, 1)) if eval_grad else -np.inf

        if not eval_grad:
            return log_likelihood

        # gradient calculation of the log-likelihood
        gamma = linalg.solve_triangular(L.T, rho).reshape(-1, 1)

        Rinv = cho_solve((L, True), np.eye(n_samples))
        Rinv_upper = Rinv[np.triu_indices(n_samples, 1)]
        _upper = gamma.dot(gamma.T)[np.triu_indices(n_samples, 1)]

        log_likelihood_grad = np.zeros((n_hyper_par, 1))

        if self.log_likelihood_mode == 'noiseless':
            # The grad tensor of R w.r.t. theta
            R_grad_tensor = self.corr_grad_theta(theta, self.X, R)

            for i in range(n_hyper_par):
                R_grad_upper = R_grad_tensor[:, :, i][np.triu_indices(n_samples, 1)]

                log_likelihood_grad[i] = np.sum(_upper * R_grad_upper) / sigma2  \
                    - np.sum(Rinv_upper * R_grad_upper)

        elif self.log_likelihood_mode == 'nugget_estim':
            # The grad tensor of R w.r.t. theta: note that the additional v below
            R_grad_tensor = alpha * self.corr_grad_theta(theta, self.X, R0)

            # partial derivatives w.r.t theta's
            for i in range(n_hyper_par-1):
                R_grad_upper = R_grad_tensor[:, :, i][np.triu_indices(n_samples, 1)]

                # Note that sigma2_noisy is used here
                log_likelihood_grad[i] = np.sum(_upper * R_grad_upper) / sigma2_total \
                    - np.sum(Rinv_upper * R_grad_upper)

            # partial derivatives w.r.t 'v'
            R_dv = R0 - np.eye(n_samples)
            log_likelihood_grad[n_hyper_par-1] = -0.5 * (np.sum(Rinv * R_dv) \
                - np.dot(gamma.T, R_dv.dot(gamma)) / sigma2_total)

        elif self.log_likelihood_mode == 'noisy':
            # TODO: implement the gradient in this case
            pass

        return log_likelihood, log_likelihood_grad

    def compute_beta_gamma(self):
        if self.beta0 is None:
            # Universal Kriging
            self.beta = linalg.solve_triangular(self.G, np.dot(self.Q.T, self.Yt))
        else:
            # Ordinary Kriging
            self.beta = np.array(self.beta0)
        self.gamma = linalg.solve_triangular(self.C.T, self.rho).reshape(-1, 1)

    def _arg_max_reduced_likelihood_function(self):
        """
        This function estimates the autocorrelation parameters theta as the
        maximizer of the reduced likelihood function.
        (Minimization of the opposite reduced likelihood function is used for
        convenience)

        Parameters
        ----------
        self : All parameters are stored in the Gaussian Process model object.

        Returns
        -------
        optimal_theta : array_like
            The best set of autocorrelation parameters (the sought maximizer of
            the reduced likelihood function).

        optimal_reduced_likelihood_function_value : double
            The optimal reduced likelihood function value.

        optimal_par : dict
            The BLUP parameters associated to thetaOpt.
        """

        # Initialize output
        best_optimal_theta = []
        best_optimal_rlf_value = []
        optimal_par = {}

        # TODO: we might not need these
        thetaL = np.atleast_2d(self.thetaL)
        thetaU = np.atleast_2d(self.thetaU)
        bounds = np.c_[thetaL.T, thetaU.T]

        if not np.isfinite(bounds).all() and self.random_start > 1:
            raise ValueError(
                    "Multiple optimizer restarts (n_restarts_optimizer>0) "
                    "requires that all bounds are finite.")

        if self.verbose:
            print("The chosen optimizer is: " + str(self.optimizer))
            if self.random_start > 1:
                print(str(self.random_start) + " random starts are required.")

        # TODO: let's make fmin_cobyla deprecated in the future!
        if self.optimizer == 'fmin_cobyla':
            percent_completed = 0.
            def minus_reduced_likelihood_function(log10t):

                return - self.log_likelihood_function(
                    hyper_par=10. ** log10t, eval_grad=False)

            if self.log_likelihood_mode == 'nugget_estim':
                alpha_bound = np.atleast_2d([1e-8, 1.0-1e-8])
                bounds = np.r_[bounds, alpha_bound]

            elif self.log_likelihood_mode == 'noisy':
                # TODO: estimation the upper and lowe bound of sigma2
                sigma2_bound = np.atleast_2d([1e-10, 1.0-1e-10])
                bounds = np.r_[bounds, sigma2_bound]

            constraints = []
            for i in range(self.theta0.size):
                constraints.append(lambda log10t, i=i:
                                   log10t[i] - np.log10(self.thetaL[0, i]))
                constraints.append(lambda log10t, i=i:
                                   np.log10(self.thetaU[0, i]) - log10t[i])

            for k in range(self.random_start):

                if k == 0:
                    # Use specified starting point as first guess
                    theta0 = self.theta0
                else:
                    # Generate a random starting point log10-uniformly
                    # distributed between bounds
                    log10theta0 = (np.log10(self.thetaL)
                                   + self.random_state.rand(*self.theta0.shape)
                                   * np.log10(self.thetaU / self.thetaL))
                    theta0 = 10. ** log10theta0

                # Run Cobyla
                try:
                    _ = np.log10(theta0).ravel()
                    log10_optimal_theta = \
                        optimize.fmin_cobyla(minus_reduced_likelihood_function,
                                             _, constraints,
                                             iprint=0)
                except ValueError as ve:
                    print("Optimization failed. Try increasing the ``nugget``")
                    raise ve

                optimal_theta = 10. ** log10_optimal_theta
                optimal_rlf_value = \
                    self.log_likelihood_function(hyper_par=optimal_theta)

                # Compare the new optimizer to the best previous one
                if k > 0:
                    if optimal_rlf_value > best_optimal_rlf_value:
                        best_optimal_rlf_value = optimal_rlf_value
                        best_optimal_theta = optimal_theta
                else:
                    best_optimal_rlf_value = optimal_rlf_value
                    best_optimal_theta = optimal_theta
                if self.verbose and self.random_start > 1:
                    if (20 * k) / self.random_start > percent_completed:
                        percent_completed = (20 * k) / self.random_start
                        print("%s completed" % (5 * percent_completed))

            optimal_rlf_value = best_optimal_rlf_value
            optimal_theta = best_optimal_theta
            optimal_rlf_value= self.log_likelihood_function(optimal_theta, optimal_par)

        elif self.optimizer == 'BFGS':
            def obj_func(log10param):
                param = 10. ** np.array(log10param)
                __ = self.log_likelihood_function(param, eval_grad=True)
                return -__[0], -__[1] * param.reshape(-1, 1)

            if self.log_likelihood_mode == 'nugget_estim':
                alpha_bound = np.atleast_2d([1e-8, 1.0-1e-8])
                bounds = np.r_[bounds, alpha_bound]

            elif self.log_likelihood_mode == 'noisy':
                # TODO: estimation the upper and lowe bound of sigma2
                sigma2_bound = np.atleast_2d([1e-10, 1.0-1e-10])
                bounds = np.r_[bounds, sigma2_bound]

            log10bounds = log10(bounds)

            llf_opt = np.inf
            dim = self.thetaL.shape[1]
            eval_budget = 500*dim
            c = 0

            # L-BFGS-B algorithm with restarts
            for iteration in range(self.random_start):
                log10param = np.random.uniform(log10bounds[:, 0], log10bounds[:, 1])
                param_opt_, llf_opt_, convergence_dict = \
                    fmin_l_bfgs_b(obj_func, log10param, pgtol=1e-5,
                                  bounds=log10bounds, maxfun=eval_budget)

                if convergence_dict["warnflag"] != 0 and self.verbose:
                    warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                          " state: %s" % convergence_dict)


                if llf_opt_ < llf_opt:
                    param_opt, llf_opt = param_opt_, llf_opt_
                    if self.verbose:
                        print('iteration: ', iteration+1, convergence_dict['funcalls'], -llf_opt)
                else:
                    c += 1

                eval_budget -= convergence_dict['funcalls']
                if eval_budget <= 0 or c > self.wait_iter:
                    break

            optimal_param = 10. ** param_opt
            optimal_llf_value = self.log_likelihood_function(optimal_param, optimal_par)

            if self.log_likelihood_mode in ['nugget_estim', 'noisy']:
                optimal_theta = optimal_param[:-1]
            else:
                optimal_theta = optimal_param

        return optimal_theta, optimal_llf_value, optimal_par


    def _check_params(self, n_samples=None):

        # Check regression model
        if not callable(self.regr):
            if self.regr in self._regression_types:
                self.regr = self._regression_types[self.regr]
            else:
                raise ValueError("regr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._regression_types.keys(), self.regr))

        # Check regression weights if given (Ordinary Kriging)
        if self.beta0 is not None:
            self.beta0 = np.atleast_2d(self.beta0)
            if self.beta0.shape[1] != 1:
                # Force to column vector
                self.beta0 = self.beta0.T

        # Check correlation model
        if not callable(self.corr):
            if self.corr in self._correlation_types:
                self.corr = self._correlation_types[self.corr]
            else:
                raise ValueError("corr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._correlation_types.keys(), self.corr))

        # Check correlation parameters
        self.theta0 = np.atleast_2d(self.theta0)
        lth = self.theta0.size

        if self.thetaL is not None and self.thetaU is not None:
            self.thetaL = np.atleast_2d(self.thetaL)
            self.thetaU = np.atleast_2d(self.thetaU)
            if self.thetaL.size != lth or self.thetaU.size != lth:
                raise ValueError("theta0, thetaL and thetaU must have the "
                                 "same length.")
            if np.any(self.thetaL <= 0) or np.any(self.thetaU < self.thetaL):
                raise ValueError("The bounds must satisfy O < thetaL <= "
                                 "thetaU.")

        elif self.thetaL is None and self.thetaU is None:
            if np.any(self.theta0 <= 0):
                raise ValueError("theta0 must be strictly positive.")

        elif self.thetaL is None or self.thetaU is None:
            raise ValueError("thetaL and thetaU should either be both or "
                             "neither specified.")

        # Force verbose type to bool
        self.verbose = bool(self.verbose)

        # Force normalize type to bool
        self.normalize = bool(self.normalize)

        # Check nugget value
        # self.nugget = np.asarray(self.nugget)
        # if np.any(self.nugget) < 0.:
        #     raise ValueError("nugget must be positive or zero.")
        # if (n_samples is not None
        #         and self.nugget.shape not in [(), (n_samples,)]):
        #     raise ValueError("nugget must be either a scalar "
        #                      "or array of length n_samples.")

        # Check optimizer
        if self.optimizer not in self._optimizer_types:
            raise ValueError("optimizer should be one of %s"
                             % self._optimizer_types)

        # Force random_start type to int
        self.random_start = int(self.random_start)