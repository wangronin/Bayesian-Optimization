import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import math
import statistics
import numpy as np
from enum import Enum


class PictureSaver:
    def __init__(self, path, name_suffix, extension):
        self.path = path
        self.fid = name_suffix
        self.extension = extension

    def save(self, fig, name):
        fig.savefig(self.path + name + '.' + self.extension)


MODE = Enum('EXECUTION MODE', 'DEBUG RELEASE')
MY_EXECUTION_MODE = MODE.RELEASE


def eprintf(*args, **kwargs):
    if MY_EXECUTION_MODE is not MODE.DEBUG:
        return
    print(*args, file=sys.stderr, **kwargs)


def fprintf(*args, **kwargs):
    with open(MY_PROGRESS_LOG_FILE, 'a') as f:
        f.write(*args, **kwargs)


class MyChartSaver:
    def __init__(self, folder_name, name, bounds, obj_function):
        if MY_EXECUTION_MODE is not MODE.DEBUG:
            return
        directory = './'+folder_name+'/'
        self.saver = PictureSaver(directory, "-"+name, "png")
        self.bounds = bounds
        self.obj_function = obj_function
        self.iter_number = 0
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.domain_grid, self.colours = MyChartSaver.__sample_points(0.1, 0.1, bounds, obj_function)

    @staticmethod
    def __sample_points(x_step, y_step, bounds, obj_f):
        x_points = MyChartSaver.__sample_on_axis(bounds[0][0], bounds[0][1], x_step)
        y_points = MyChartSaver.__sample_on_axis(bounds[1][0], bounds[1][1], y_step)
        v = []
        N = len(x_points)*len(y_points)
        eprintf("Number of points on the grid is", N)
        points = [[0. for _ in range(N)] for _ in range(2)]
        values = [0.] * N
        cnt = 0
        for x in x_points:
             for y in y_points:
                points[0][cnt] = x
                points[1][cnt] = y
                values[cnt] = obj_f([x, y])
                cnt += 1
        return points, MyChartSaver.__compute_colours_2(values)

    @staticmethod
    def __sample_on_axis(beg, end, step):
        N = int(math.ceil((end - beg) / step))
        ans = [0.] * N
        x0 = beg
        cnt = 0
        while cnt < N:
            ans[cnt] = x0
            x0 += step
            cnt += 1
        return ans

    @staticmethod
    def __compute_colours_2(Y):
        colours = []
        y_copy = Y.copy()
        y_copy.sort()
        min_value = y_copy[0]
        k = int(0.4 * len(Y))
        m = math.log(0.5) / (y_copy[k] - min_value)
        jet_cmap = mpl.cm.get_cmap(name='jet')
        for y in Y:
            colours.append(jet_cmap(1. - math.exp(m * (y - min_value))))
        return colours

    @staticmethod
    def __get_column_variances(X):
        variances = []
        for i in range(len(X[0])):
            xi = statistics.variance(X[:, i])
            variances.append(xi)
        return variances

    @staticmethod
    def __get_sorted_var_columns_pairs(X):
        var = MyChartSaver.__get_column_variances(X)
        all_var = sum(vi ** 2 for vi in var)
        var_col = [(var[i], i) for i in range(len(var))]
        var_col.sort()
        var_col.reverse()
        return var_col

    def set_iter_number(self, iter_number):
        if MY_EXECUTION_MODE is not MODE.DEBUG:
            return
        self.iter_number = iter_number

    def create_figure_with_domain(self):
        if MY_EXECUTION_MODE is not MODE.DEBUG:
            return
        fig = plt.figure()
        plt.xlim(list(self.bounds[0]))
        plt.ylim(list(self.bounds[1]))
        plt.scatter(self.domain_grid[0], self.domain_grid[1], marker='s', c=self.colours)
        return fig

    def add_evaluated_points(self, iter_number, X):
        if MY_EXECUTION_MODE is not MODE.DEBUG:
            return
        plt.title(f'Iteration number {iter_number}, last point is ({X[-1][0]:.4f}, {X[-1][1]:.4f})')
        plt.scatter(X[:-1, 0], X[:-1, 1], c='black', marker='X')
        plt.scatter(X[-1][0], X[-1][1], c='red', marker='X')

    def add_mainfold(self, X_transformed, inverser):
        if MY_EXECUTION_MODE is not MODE.DEBUG:
            return
        X = inverser.inverse_transform(X_transformed)
        plt.scatter(X[:, 0], X[:, 1], c='green')

    def save(self, iter_number, X):
        if MY_EXECUTION_MODE is not MODE.DEBUG:
            return
        fig = self.create_figure_with_domain()
        self.add_evaluated_points(iter_number, X)
        self.saver.save(fig, f"DoE-{iter_number}")

    def save_with_manifold(self, iter_number, X, X_transformed, lb_f, ub_f, inverser):
        if MY_EXECUTION_MODE is not MODE.DEBUG:
            return
        if len(X_transformed[0]) > 1:
            return
        fig = self.create_figure_with_domain()
        self.add_mainfold(X_transformed, inverser)
        self.add_evaluated_points(iter_number, X)
        N = 100
        start = lb_f
        step = (ub_f - lb_f) / N
        extra_points = []
        for i in range(N):
            extra_points.append([start + i*step])
        # extra_points = np.array(extra_points).transpose()
        self.add_mainfold(extra_points, inverser)
        self.saver.save(fig, f"DoE-{iter_number}")

    def save_feature_space(self, X, y):
        if MY_EXECUTION_MODE is not MODE.DEBUG:
            return
        fig = plt.figure()
        colors = MyChartSaver.__compute_colours_2(y)
        plt.title(f'Iteration number {self.iter_number}, last point is ({X[-1][0]:.4f}, {X[-1][1]:.4f})')
        plt.scatter(X[:,0], X[:,1], c=colors)
        self.saver.save(fig, f"Feature-Space-{self.iter_number}")

    def save_variances(self, X):
        if MY_EXECUTION_MODE is not MODE.DEBUG:
            return
        fig = plt.figure()
        var = self.__get_sorted_var_columns_pairs(X)
        plt.bar([i for i in range(len(var))], [a for (a,b) in var])
        plt.gca().set_xticks([i for i in range(len(var))])
        plt.gca().set_xticklabels([str(b) for (a,b) in var])
        plt.ylabel("$ {\sigma^2_i}/{\sum \sigma^2_i}$")
        plt.xlabel("$\sigma^2_i$")
        plt.title(f'Iteration number {self.iter_number}')
        self.saver.save(fig, f'Variance-{self.iter_number}')

    def save_model(self, model, X, y_):
        if MY_EXECUTION_MODE is not MODE.DEBUG:
            return
        if len(X[0]) > 1:
            fig = plt.figure()
            plt.title(f'Model function after iteration {self.iter_number} has dimensionality {len(X[0])}')
            self.saver.save(fig, f'Model-{self.iter_number}')
            return
        N = 500
        fig = plt.figure()
        X_ = np.linspace(X[:,0].min(), X[:,0].max(), N)
        X_ = np.concatenate((X_, np.array([x[0] for x in X])), axis=0)
        X_.sort()
        Y_predicted = model.predict(np.array([[x] for x in X_]))
        Y_predicted = [y[0] for y in Y_predicted]
        plt.plot(X_, Y_predicted)
        plt.title(f'Model function after iteration {self.iter_number}')
        plt.scatter(X, y_, c='red')
        self.saver.save(fig, f'Model-{self.iter_number}')
        # eprintf("Length scale", model.my_internal_kernel.length_scale)
        # eprintf(f'Iteration number {self.iter_number}')
        # eprintf("Features space points\n", X)
        # eprintf("Scaled function values\n", y_)
        # eprintf("Model predicts\n", [y[0] for y in model.predict(X)])

