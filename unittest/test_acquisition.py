# TODO: diagnostic plot for the gradient of Infill-Criteria
# goes to unittest
from GaussianProcess.trend import linear_trend, constant_trend
from GaussianProcess import GaussianProcess
from GaussianProcess.utils import plot_contour_gradient
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from deap import benchmarks

np.random.seed(123)

plt.ioff()
fig_width = 16
fig_height = 16

noise_var = 0.
def fitness(X):
    X = np.atleast_2d(X)
    return np.array([benchmarks.schwefel(x)[0] for x in X]) + \
        np.sqrt(noise_var) * np.random.randn(X.shape[0])
    
dim = 2
n_init_sample = 10

x_lb = np.array([-5] * dim)
x_ub = np.array([5] * dim)

X = np.random.rand(n_init_sample, dim) * (x_ub - x_lb) + x_lb
y = fitness(X)

thetaL = 1e-5 * (x_ub - x_lb) * np.ones(dim)
thetaU = 10 * (x_ub - x_lb) * np.ones(dim)
theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

mean = linear_trend(dim, beta=None)
model = GaussianProcess(mean=mean, corr='matern', theta0=theta0, thetaL=thetaL, thetaU=thetaU,
                        nugget=None, noise_estim=True, optimizer='BFGS', verbose=True,
                        wait_iter=3, random_start=10, eval_budget=50)

model.fit(X, y)

def grad(model):
    f = MGFI(model, t=10)
    def __(x):
        _, dx = f(x, dx=True)
        return dx
    return __

t = 1
infill = MGFI(model, t=t)
infill_dx = grad(model)

m = lambda x: model.predict(x)
sd2 = lambda x: model.predict(x, eval_MSE=True)[1]

m_dx = lambda x: model.gradient(x)[0]
sd2_dx = lambda x: model.gradient(x)[1]

if 1 < 2:
    fig0, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=False, sharex=False,
                                figsize=(fig_width, fig_height),
                                subplot_kw={'aspect': 'equal'}, dpi=100)
                                
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

    plot_contour_gradient(ax0, fitness, None, x_lb, x_ub, title='Noisy function',
                            n_level=20, n_per_axis=200)
    
    plot_contour_gradient(ax1, m, m_dx, x_lb, x_ub, title='GPR estimation',
                            n_level=20, n_per_axis=200)
                            
    plot_contour_gradient(ax2, sd2, sd2_dx, x_lb, x_ub, title='GPR variance',
                            n_level=20, n_per_axis=200)
    plt.tight_layout()

fig1, ax3 = plt.subplots(1, 1, figsize=(fig_width, fig_height),
                            subplot_kw={'aspect': 'equal'}, dpi=100)
                            
plot_contour_gradient(ax3, infill, infill_dx, x_lb, x_ub, title='Infill-Criterion',
                        is_log=True, n_level=50, n_per_axis=250)

plt.tight_layout()
plt.show()
