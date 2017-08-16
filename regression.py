# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:05:01 2017

@author: wangronin
"""

from __future__ import division   # important! for the float division

import pdb
import sys, os, time, string, random
from copy import deepcopy
from datetime import datetime

import numpy as np
from pandas import DataFrame


def fitting(**kwargs):
    """
    Function that calls the actual model fitting and calculate the performance metrics
    """
    # risk functions
    from sklearn.metrics import r2_score, mean_squared_error

    # if this function is called from MPI master process
    if not bool(kwargs):

        from mpi4py import MPI
        comm = MPI.Comm.Get_parent()

        model = comm.scatter(None, root=0)
        data = comm.scatter(None, root=0)

        index, training, test = data
        test_set, target = test

        # Model fitting in parallel...
        model.fit(*training)

        # Model predictions
        train_pred = model.predict(training[0])
        test_pred = model.predict(test_set)

        # Compute the performance measures
        mse = mean_squared_error(target, test_pred)
        r2 = r2_score(target, test_pred)

        # Synchronization...
        comm.Barrier()

        # Gathering the fitted kriging model back
        fitted = {
                  'index': index,  # the fold index
                  'model': model,
                  'r2': r2,
                  'mse': mse,
                  'y_pre_train': train_pred,
                  'y_pre_test': test_pred
                  }

        comm.gather(fitted, root=0)
        comm.Disconnect()

    else:
        model, training, test = map(kwargs.get, ('model', 'training', 'test'))
        model_ = deepcopy(model)
        model_.fit(*training)

        # Model predictions
        test_set, target = test
        train_pred = model_.predict(training[0])
        test_pred = model_.predict(test_set)

        # Compute the performance measures
        mse = mean_squared_error(target, test_pred)
        r2 = r2_score(target, test_pred)

        return model_, r2, mse, train_pred, test_pred


if len(sys.argv) == 2 and sys.argv[1] == '-slave':
    fitting()
    sys.exit()


class regressor:

    def __init__(self, algorithm='SVM', metric='MSE', model_par=None,
                 light_mode=False, tmp=False, verbose=True, random_seed=None):

        # to include the most commonly used regression models:
        # SVM, random forest, Gaussian Process
        # TODO: add neural network, RBF network
        self._supported_algorithm = ['GPR', 'SVM', 'RF', 'tree']
        self.verbose = verbose
        self.metric = metric
        self.light_mode = light_mode
        self.tmp = tmp
        self.random_seed = random_seed

        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # generate a 10-letter model ID
        self.ID = ''.join(random.choice(string.ascii_letters + string.digits) \
                          for _ in range(10))

        if algorithm not in self._supported_algorithm:
            raise Exception('not supported algorithm!')
        else:
            self.algorithm = algorithm

        # intake the model parameter
        self.model_par = model_par
        if self.algorithm == 'SVM':
            pass
            # the configurable parameters for SVM
#            self.model_conf_par = ['C', 'epsilon', 'gamma']
#            self.mode_conf_par_lb = [1e-10, 1e-10, 1e-10]
#            self.mode_conf_par_ub = [10, 0.9, 0.9]

    def __digest_par(self):
        # TODO: setup model's parameters based on input
        pass

    def __preprocessing(self, X, y, normalize=True):
        # basic data checking and transformation that is only valid for continuous input

        # data dimensionality checking
        self.X = np.atleast_2d(X)
        self.y = np.array(y)
        self.n_sample = len(self.y)

        self.X = self.X.T if self.X.shape[0] != self.n_sample else self.X
        assert self.X.shape[0] == self.n_sample

        self.n_feature = self.X.shape[1]

        # Normalize (centering) the input and output
        if normalize:
            self.X_mean = np.mean(self.X, axis=0)
            self.X_std = np.std(self.X, axis=0)
            self.y_mean = np.mean(self.X)
            self.y_std = np.std(self.X)

            self.X_ = (self.X - self.X_mean) / self.X_std
            self.y_ = (self.y - self.y_mean) / self.y_std
        else:
            self.X_, self.y_ = self.X, self.y


    def __create_model(self):
        # model creation
        # TODO: allow for the configuration/tuning of the hyper/meta-parameters below
        if self.algorithm == 'GPR':
            from owck import OWCK

            thetaL = 1e-5 * np.ones(self.n_feature)
            thetaU = 1e2 * np.ones(self.n_feature)
            theta0 = theta0 = np.random.rand(self.n_feature) * (thetaU - thetaL) + thetaL

            self.model = OWCK(corr='matern', n_cluster=None, min_leaf_size=50,
                              cluster_method='tree',
                              theta0=theta0, thetaL=thetaL, thetaU=thetaU, nugget=1e-10,
                              random_start=1, optimizer='BFGS', nugget_estim=True,
                              normalize=False,
                              is_parallel=False)

        elif self.algorithm == 'SVM':
            from sklearn.svm import SVR
            self.model = SVR(cache_size=200, coef0=0.0, kernel='rbf', max_iter=-1,
                             shrinking=True, tol=0.001, verbose=False, **self.model_par)

            # the configurable parameters for SVM
            self.model_conf_par = ['C', 'epsilon', 'gamma']
            self.mode_conf_par_lb = [1e-10, 1e-10, 1e-10]
            self.mode_conf_par_ub = [10, 0.9, 0.9]

        elif self.algorithm == 'RF':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=50, max_depth=None,
                                               min_samples_split=10,
                                               min_samples_leaf=5)

        elif self.algorithm == 'tree':
            from sklearn.tree import DecisionTreeRegressor
            self.model = DecisionTreeRegressor(max_depth=5)

    def __export_CV_info(self, K, train_idx, test_idx, y_true, y_pred, score,
                         output='csv', path=None):
        index = np.r_[train_idx, test_idx]
        is_test = np.r_[np.zeros(train_idx.shape), np.ones(test_idx.shape)]

        df = DataFrame(np.c_[index, is_test, y_true, y_pred],
                       columns=['index', 'is_test', 'y_true', 'y_pred'])

        # if no path provided, save the current working dir...
        if path is None:
            path = os.path.join(os.getcwd(), 'tmp')

        savetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = os.path.join(path, self.ID + '-Fold{}-{}.csv'.format(K, savetime))
        df.to_csv(filename, index=False)

        filename = os.path.join(path, self.ID + '-Fold{}score-{}.csv'.format(K, savetime))
        score.to_csv(filename, index=False)

    def fit(self, X, y, n_fold=10, parallel=False):

        # Generate the model
        # perform the mode cross-validation
        self.__preprocessing(X, y)
        self.__create_model()

        if n_fold == 0:
            print 'No cross-validation specified...'

            self.model, r2, mse, __ = fitting(model=self.model, training=(self.X_,
                                                                          self.y_),
                                          test=(self.X_, self.y_))

            print 'Model fitting done...'

        else:
            # generate kFold info...
            from sklearn.model_selection import KFold

            skf = KFold(n_splits=n_fold, shuffle=True,
                        random_state=self.random_seed)
            index = [(train_index, test_index) \
                     for train_index, test_index in skf.split(X)]

            training_set = [(X[train_index, :], y[train_index]) \
                for train_index, test_index in index]
            test_set = [(X[test_index, :], y[test_index]) \
                for train_index, test_index in index]

            t_start = time.time()

            if parallel: # use MPI to parallelize the CV
                from mpi4py import MPI
                comm = MPI.COMM_WORLD

                models = [deepcopy(self.model) for i in range(n_fold)]
                comm = MPI.COMM_SELF.Spawn(sys.executable,
                                           args=['./regression.py', '-slave'],
                                           maxprocs=n_fold)

                # scatter the models and data
                comm.scatter(models, root=MPI.ROOT)
                comm.scatter([(k, training_set[k], test_set[k]) for k in range(n_fold)],
                              root=MPI.ROOT)

                # Synchronization while the children process are performing
                # heavy computations...
                comm.Barrier()

                # Gether the fitted model from the childrenn process
                # Note that 'None' is only valid in master-slave working mode
                results = comm.gather(None, root=MPI.ROOT)

                # free all slave processes
                comm.Disconnect()

                # tic toc
                t_stop = time.time()
                self.models = [results[i]['model'] for i in range(n_fold)]

                # register the measures
                scores = DataFrame([[d['r2'], d['mse']] for d in results],
                                   columns=['r2', 'mse'])
                scores.index = [d['index'] for d in results]
                scores.sort_index('index', inplace=True)

                if self.verbose:
                    print 'training takes {0:.3f}mins'.format((t_stop - t_start)/60)
                    print 'model performance:'
                    print scores
                    print

                # save CV information to csv...
                for i in range(n_fold):
                    y_pre_test = results[i]['y_pre_test']
                    y_pre_train = results[i]['y_pre_train']

                    if self.tmp:
                        self.__export_CV_info(i+1, index[i][0], index[i][1],
                                          np.r_[training_set[i][1], test_set[i][1]],
                                          np.r_[y_pre_train, y_pre_test], score=scores)

            else:  # sequential execution
                self.models = [] # to store the fitted models
                r2 = []
                mse = []
                for i in range(n_fold):

                    if self.verbose:
                        print 'fitting on fold {}/{}...'.format(i+1, n_fold)

                    _, r2_, mse_, __, ___ = fitting(model=self.model,
                                                    training=training_set[i],
                                                    test=test_set[i])
                    r2.append(r2_)
                    mse.append(mse_)
                    y_pre_train, y_pre_test = __, ___
                    self.models.append(_)
#                    self.__export_CV_info(i+1, index[i][0], index[i][1],
#                                          np.r_[training_set[i][1], test_set[i][1]],
#                                          np.r_[y_pre_train, y_pre_test], score=scores)
                    if self.verbose:
                        print 'r2: {}'.format(r2_)
                        print 'MSE: {}'.format(mse_)

                scores = DataFrame(np.c_[r2, mse], columns=['r2', 'mse'])
                scores.index = range(n_fold)

            self.performance = scores

            if self.light_mode:
                del self.X, self.X_, self.y, self.y_
                del self.X_mean, self.X_std, self.y_mean, self.y_std


    def visualize2D(self, x_lb, x_ub, ax=None):
        pass

    def predict(self, X):
        if hasattr(self, 'models'):
            # TODO: to verify the heuristic: using the models from Kfold-CV
            # as an ensemble...

            __ = [model.predict(X) for model in self.models]
            # TODO: compare to the median, which should be better
            y_pred = np.mean(__)

        else:
            y_pred = self.model.predict(X)

        return y_pred

    def dump(self, path=None):
        from cPickle import dump

        # if no path provided, save the current working dir...
        if path is None:
            path = os.getcwd()

        filename = self.ID + '-' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.pkl'
        file_path = os.path.join(path, filename)
        f = file(file_path, 'w')
        dump(self, f)  # Save the model to pickle file
        f.close()

if __name__ == '__main__':
    np.random.seed(1)
    from fitness import rastrigin

    # test problem: to fit a so-called Rastrigin function in 20D
    X = np.random.rand(500, 20)
    y = rastrigin(X.T)

    model = regressor(algorithm='SVM', metric='mse',
                      model_par={'epsilon': 0.30000000000000004,
                                 'C': 14.000000000000002,
                                 'gamma': 1.4000000000000001},
                      verbose=True)

    model.fit(X, y, n_fold=3, parallel=True)


## data plotting
#fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
#
#acc_data = [acc_rate[i, :, :].flatten() for i in range(n_model)]
#auc_data = [auc_score[i, :, :].flatten() for i in range(n_model)]
#mse_data = [mse_score[i, :, :].flatten() for i in range(n_model)]
#
#axes[0].boxplot(acc_data)
#axes[1].boxplot(mse_data)
#axes[2].boxplot(auc_data)
#
#axes[0].set_title('accuracy')
#axes[1].set_title('MSE')
#axes[2].set_title('AUC')
#
#for ax in axes:
#    ax.grid(True)
#    ax.hold(True)
#    ax.set_xticklabels(model_list)
#
#fig.savefig(fig_path + '/cv-comparison.pdf')

#fig, axes = plt.subplots(2, 5, figsize=(fig_width, fig_height), dpi=100)
#axes.shape = (n_rep, )
#
#for k, ax in enumerate(axes):
#    ax.grid(True)
#    ax.hold(True)
#
#    mean_tpr = 0.0
#    mean_fpr = np.linspace(0, 1, 100)
#    for n in range(n_folds):
#        fpr = rate[k][n][:, 0]
#        tpr = rate[k][n][:, 1]
#        roc_auc = auc_score[i, n, j]
#
#        mean_tpr += interp(mean_fpr, fpr, tpr)
#        mean_tpr[0] = 0.0
#
#        ax.plot(fpr, tpr, lw=1, color = _color[n],
#                label='ROC fold %d (area = %0.2f)' % (n+1, roc_auc))
#
#    mean_tpr /= n_folds
#    mean_tpr[-1] = 1.0
#    mean_auc = auc(mean_fpr, mean_tpr)
#
#    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='random gusess')
#    ax.plot(mean_fpr, mean_tpr, 'k--',
#            label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
#    ax.set_xlim([-0.05, 1.05])
#    ax.set_ylim([-0.05, 1.05])
#    ax.set_title('Rep {}'.format(k+1))
#    ax.legend(loc="lower right", prop={'size': 10})
#
#fig.text(0.5, 0.04, 'False Positive Rate', ha='center', va='center',
#         fontsize=17)
#fig.text(0.1, 0.5, 'True Positive Rate', ha='center',
#         va='center', rotation='vertical', fontsize=17)
#fig.suptitle("ROC of model {}".format(model))
#fig.savefig(fig_path + '/ROC-{}.pdf'.format(model))

