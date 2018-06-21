# -*- coding: utf-8 -*-

# Author: Hao Wang <wangronin@gmail.com>
#         Bas van Stein <bas9112@gmail.com>

"""
Created on Wed Dec 16 16:34:04 2015

@author: wangronin
"""

import warnings
import numpy as np
from numpy import dot, sqrt, array, zeros, ones, exp

from scipy.linalg import solve_triangular, inv
from scipy import linalg

from sklearn.gaussian_process import correlation_models as correlation
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import manhattan_distances

from .gpr import GaussianProcess

MACHINE_EPSILON = np.finfo(np.double).eps

def my_dot(x, y):
    n_row = x.shape[0]
    n_col = y.shape[1]
    res = np.zeros((n_row, n_col))
    for i in range(n_row):
        for j in range(n_col):
            res[i, j] = np.sum(x[i, :] * y[:, j])
    return res


class GaussianProcess_extra(GaussianProcess):

    def __init__(self, regr='constant', corr='squared_exponential',
                 beta0=None, verbose=False, theta0=1e-1, thetaL=None, thetaU=None, sigma2=None,
                 optimizer='BFGS', random_start=10, wait_iter=5, likelihood='restricted',
                 eval_budget=None, nugget=10. * MACHINE_EPSILON, nugget_estim=False, 
                 random_state=None):

        super(GaussianProcess_extra, self).__init__(regr=regr, corr=corr, beta0=beta0,
                 verbose=verbose, theta0=theta0, thetaL=thetaL, thetaU=thetaU,
                 optimizer=optimizer, random_start=random_start, likelihood=likelihood,
                 nugget=nugget, nugget_estim=nugget_estim, wait_iter=wait_iter, 
                 eval_budget=eval_budget, random_state=random_state)

    def gradient(self, x):
        """
        Calculate the gradient of the posterior mean and variance
        Note that the nugget effect will not the change the computation below
        """

        check_is_fitted(self, 'X')

        # Check input shapes
        x = np.atleast_2d(x)
        n_eval, _ = x.shape
        n_samples, n_features = self.X.shape

        if _ != n_features:
            raise Exception('x does not have the right size!')

        if n_eval != 1:
            raise Exception('x must be a vector!')

        # trend and its Jacobian
        f = self.mean.F(x)
        f_dx = self.mean.Jacobian(x)

        # correlation and its Jacobian
        d = manhattan_distances(x, Y=self.X, sum_over_features=False)
        r = self.corr(self.theta_, d).reshape(n_eval, n_samples)
        r_dx = self.corr_dx(x, X=self.X, r=r)

        # gradient of the posterior mean
        y_dx = dot(f_dx, self.beta) + my_dot(r_dx, self.gamma)

        # auxiliary variable: rt = C^-1 * r
        rt = solve_triangular(self.C, r.T, lower=True)
        rt_dx = solve_triangular(self.C, r_dx.T, lower=True).T

        # auxiliary variable: u = Ft^T * rt - f
        u = dot(self.Ft.T, rt) - f
        u_dx = dot(rt_dx, self.Ft) - f_dx

        mse_dx = -dot(rt_dx, rt)      # for Simple Kriging
        if self.beta0 is None:        # for Universal Kriging
            Ft2inv = inv(dot(self.Ft.T, self.Ft))
            mse_dx += dot(u_dx, Ft2inv).dot(u)

        mse_dx = 2.0 * self.sigma2 * mse_dx

        return y_dx, mse_dx

    def corr_dx(self, x, X=None, theta=None, r=None, nu=1.5):
        # Check input shapes
        x = np.atleast_2d(x)
        n_eval, _ = x.shape
        n_samples, n_features = self.X.shape

        if _ != n_features:
            raise Exception('x does not have the right sizeh!')

        if n_eval != 1:
            raise Exception('x must be a vector!')

        if self.theta_ is None:
            raise Exception('The model is not fitted yet!')

        X = np.atleast_2d(X)
        if X is None:
            X = self.X

        diff = (x - X).T

        if theta is None:
            theta = self.theta_

        # calculate the required variables if not given
        if r is None:
            r = self.corr(self.theta_, np.abs(diff).T)

        theta = theta.reshape(-1, 1)

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                if self.corr_type == 'squared_exponential':
                    grad = -2 * r * (theta * diff)

                elif self.corr_type == 'matern':
                    c = np.sqrt(3)
                    D = np.sqrt(np.sum(theta * diff ** 2., axis=0))

                    if nu == 0.5:
                        grad = - diff * theta / D * r
                    elif nu == 1.5:
                        grad = diff * theta / D
                        grad *= -3. * D * exp(-c * D)
                    elif nu == 2.5:
                        pass

                elif self.corr_type == 'absolute_exponential':
                    grad = -r * theta * np.sign(diff)
                elif self.corr_type == 'generalized_exponential':
                    pass
                elif self.corr_type == 'cubic':
                    pass
                elif self.corr_type == 'linear':
                    pass
            except Warning:
                grad = np.zeros((n_features, n_samples))

        return grad

    def prior_cov_matrix(self, X, corr=False, normalize=False):
        # Check input shapes
        X = np.atleast_2d(X)

        n_eval, _ = X.shape
        n_samples, n_features = self.X.shape
        n_samples_y, n_targets = self.y.shape

        if X.shape[1] != n_features:
            raise ValueError(("The number of features in X (X.shape[1] = %d) "
                              "should match the number of features used "
                              "for fit() "
                              "which is %d.") % (X.shape[1], n_features))

        # Do not to normalize the inputs first!
        if normalize:
            X = (X - self.X_mean) / self.X_std

        dx_new = manhattan_distances(X, Y=X, sum_over_features=False)

        # Do not miss the error/noise term: self.nugget
        R_prior = self.corr(self.theta_, dx_new).reshape(n_eval, n_eval)

        if correlation:
            return R_prior

        else:
            C_prior = array([self.sigma2[i] * R_prior for i in range(n_targets)])
            C_prior = sqrt((C_prior ** 2.).sum(axis=0) / n_targets) + \
                np.diag(self.nugget * ones(n_eval))

            return C_prior

    def prior_cov_Mat1Mat2(self, X1, X2, corr=False, normalize=False):
        # Check input shapes
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        n_eval_X1, _ = X1.shape
        n_eval_X2, _ = X2.shape
        n_samples, n_features = self.X.shape
        n_samples_y, n_targets = self.y.shape

        if X1.shape[1] != n_features or X2.shape[1] != n_features:
            raise ValueError(("The number of features in X (X.shape[1] = %d) "
                              "should match the number of features used "
                              "for fit() "
                              "which is %d.") % (X1.shape[1], n_features))

        # remember to normalize the inputs
        if normalize:
            X1 = (X1 - self.X_mean) / self.X_std
            X2 = (X2 - self.X_mean) / self.X_std

        dx_new = manhattan_distances(X1, Y=X2, sum_over_features=False)
        R_prior = self.corr(self.theta_, dx_new).reshape(n_eval_X1, n_eval_X2)

        if corr:
            return R_prior
        else:
            C_prior = array([self.sigma2[i] * R_prior for i in range(n_targets)])
            C_prior = sqrt((C_prior ** 2.).sum(axis=0) / n_targets)

            return C_prior
    
    def posterior_kernel(self):
        # TODO: 
        # create the posterior kernel function
        return None

    def predict(self, X, eval_MSE=False, eval_cov=False, batch_size=None):
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
            Default assumes eval_MSE = False and evaluates only the BLUP (mean
            prediction).

        eval_cov : boolean, option
            A boolean specifying whether the Posterior Covariance Matrix should
            be evaluated or not.
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
        X = np.atleast_2d(X)
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
            f = self.mean.F(X)
            r = self.corr(self.theta_, dx).reshape(n_eval, n_samples)

            if eval_cov:
                R_cross = self.prior_cov_matrix(X, correlation=True)

            # Scaled predictor
            y_ = dot(f, self.beta) + my_dot(r, self.gamma)

            # Predictor
            y = (self.y_mean + self.y_std * y_).reshape(n_eval, n_targets)

            if self.y_ndim_ == 1:
                y = y.ravel()

            # Mean Squared Error
            if eval_MSE or eval_cov:
                C = self.C
                if C is None:
                    # Light storage mode (need to recompute C, F, Ft and G)
                    if self.verbose:
                        print("This GaussianProcess used 'light' storage mode "
                              "at instantiation. Need to recompute "
                              "autocorrelation matrix...")
                    self.reduced_likelihood_function_value, par = \
                        self.reduced_likelihood_function()
                    self.C = par['C']
                    self.Ft = par['Ft']
                    self.G = par['G']

                rt = linalg.solve_triangular(self.C, r.T, lower=True)

                if self.beta0 is None:
                    # Universal Kriging
                    u = linalg.solve_triangular(self.G.T,
                                                dot(self.Ft.T, rt) - f.T,
                                                lower=True)
                else:
                    # Ordinary Kriging
                    u = zeros((n_targets, n_eval))

                if eval_MSE:
                    MSE = dot(self.sigma2.reshape(n_targets, 1),
                              (1. - (rt ** 2.).sum(axis=0)
                              + (u ** 2.).sum(axis=0))[np.newaxis, :])
                    MSE = np.sqrt((MSE ** 2.).sum(axis=0) / n_targets)

                    # Mean Squared Error might be slightly negative depending on
                    # machine precision: force to zero!
                    MSE[MSE < 0.] = 0.

                    if self.y_ndim_ == 1:
                        MSE = MSE.ravel()

                # TODO: verify this...
                # Compute the posterior covariance matrix
                if eval_cov:
                    poster_cov = (R_cross - dot(rt.T, rt) + dot(u.T, u))
                    poster_cov = array([self.sigma2[i] * poster_cov \
                        for i in range(n_targets)])
                    poster_cov = sqrt((poster_cov ** 2.).sum(axis=0) / n_targets)

                res = [y]
                res += [MSE] if eval_MSE else []
                res += [poster_cov] if eval_cov else []

                return tuple(res)

            else:

                return y

        else:
            # Memory management
            if type(batch_size) is not int or batch_size <= 0:
                raise Exception("batch_size must be a positive integer")

            if eval_cov:
                raise Exception("batch mode is not infeasible for " +
                    "posterior covariance computation")

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


# if __name__ == '__main__':

#     # diagostic plots of gradient field
#     def plot_contour_gradient(ax, f, grad, x_lb, x_ub, title='f', n_per_axis=200):

#         from numpy import linspace, meshgrid
#         import matplotlib.colors as colors
#         fig = ax.figure

#         x = linspace(x_lb[0], x_ub[0], n_per_axis)
#         y = linspace(x_lb[1], x_ub[1], n_per_axis)
#         X, Y = meshgrid(x, y)

#         fitness = array([f(p) for p in np.c_[X.flatten(), Y.flatten()]]).reshape(-1, len(x))
#         ax.contour(X, Y, fitness, 25, colors='k', linewidths=1)

#         # calculate function gradients
#         x1 = linspace(x_lb[0], x_ub[0], np.floor(n_per_axis / 10))
#         x2 = linspace(x_lb[1], x_ub[1], np.floor(n_per_axis / 10))
#         X1, X2 = meshgrid(x1, x2)

#         dx = array([grad(p).flatten() for p in np.c_[X1.flatten(), X2.flatten()]])
#         dx_norm = np.sqrt(np.sum(dx ** 2.0, axis=1))
#         dx /= dx_norm.reshape(-1, 1)
#         dx1 = dx[:, 0].reshape(-1, len(x1))
#         dx2 = dx[:, 1].reshape(-1, len(x1))

#         CS = ax.quiver(X1, X2, dx1, dx2, dx_norm, cmap=plt.cm.gist_rainbow,
#                         norm=colors.LogNorm(vmin=dx_norm.min(), vmax=dx_norm.max()),
#                         headlength=5)

#         fig.colorbar(CS, ax=ax, fraction=0.046, pad=0.04)

#         ax.set_xlabel('$x_1$')
#         ax.set_ylabel('$x_2$')
#         ax.grid(True)
#         ax.set_title(title, y=1.05)
#         ax.set_xlim(x_lb[0], x_ub[0])
#         ax.set_ylim(x_lb[1], x_ub[1])

#     import matplotlib.pyplot as plt

#     np.random.seed(100)

#     fig_width = 22
#     fig_height = fig_width * 9 / 16

#     n_sample = 10
#     X = np.random.rand(n_sample, 2) * n_sample - 5
#     y = np.sin(np.sum(X, axis=1) ** 2.0)

#     model = GaussianProcess_extra(corr='matern', theta0=[1e-1, 5*1e-1], normalize=False)

#     model.fit(X, y)

#     # To verify the corr gradient computation
#     x_lb = [-5, -5]
#     x_ub = [5, 5]

#     # Plot EI and PI landscape
#     fig0, (ax0, ax1) = plt.subplots(1, 2, figsize=(fig_width, fig_height),
#                                     subplot_kw={'aspect':'equal'}, dpi=100)
#     fig0.subplots_adjust(left=0.03, bottom=0.01, right=0.97, top=0.99, wspace=0.08,
#                         hspace=0.1)

#     ax0.set_xlim([x_lb[0], x_ub[0]])
#     ax0.set_ylim([x_lb[1], x_ub[1]])
#     ax1.set_xlim([x_lb[0], x_ub[0]])
#     ax1.set_ylim([x_lb[1], x_ub[1]])

#     f = lambda x: model.predict(x)
#     grad = lambda x: model.gradient(x)[0]

#     plot_contour_gradient(ax0, f, grad, x_lb, x_ub, title='', n_per_axis=200)

#     f = lambda x: model.predict(x, eval_MSE=True)[1]
#     grad = lambda x: model.gradient(x)[1]

#     plot_contour_gradient(ax1, f, grad, x_lb, x_ub, title='', n_per_axis=200)


#     plt.show()
