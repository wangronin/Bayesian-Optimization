"""
Created on Mon Aug 10 15:05:01 2015

@author: wangronin
"""


from __future__ import division   # important! for the float division

import pdb
import time
import sys
import os

import numpy as np
from numpy import array, c_, mean, zeros, ones

from scipy import interp

from rpy2.robjects import r
from rpy2.robjects import pandas2ri

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import auc

from gpc import GPC

from pandas import DataFrame

import cPickle as cp

from svc_tune import SVC_tune

# Try to import MPI module
try:

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_process = comm.Get_size()

except:
    raise Exception('No module named mpi4py!')

pandas2ri.activate()

# plot settings
plt.ioff()
fig_width = 22
fig_height = 22 * 9 / 16
_color = ['r', 'm', 'b', 'g', 'c']

rcParams['font.size'] = 15

# setup working directory
os.chdir(os.path.expanduser('~')  + '/Desktop/PROMIMOOC/BMW')

def load_data_from_file(src):
    pass


class classifier:

    def __init__(self, algorithm = 'SVM'):

        self._algorithms = ['naive bayes', 'logit', 'decison tree', 'random forest']

        if algorithm not in self._algorithms:
            raise Exception('not supported algorithm!')
        else:
            self.algorithm = algorithm


    def __tuner(self):
        pass

    def train(data, target = None):

        # Generate the model
        if model_type == 'svm':

            # perform grid search to tune the SVM parameters
            C = np.logspace(2**-5, 2**15, 10, base=2)
            gamma = np.logspace(2**-15, 2**3, 10, base=2)

            models = [SVC_tune(C, gamma, is_parallel=True,
                               kernel='rbf',
                               maxprocs = 10*16/n_folds,
                               class_weight='auto') \
                for k in range(n_folds)]

            elif model_type == 'naive bayes':
                models = [naive_bayes.BernoulliNB() for k in range(n_folds)]

            elif model_type == 'random forest':
                models = [RandomForestClassifier(n_estimators=500,
                                                 class_weight='auto') \
                                                 for k in range(n_folds)]

            elif model_type == 'logit':
                models = [LogisticRegression(class_weight='auto', max_iter=200,
                                             solver='lbfgs') \
                                             for k in range(n_folds)]

            elif model_type == 'gpc':

                dim = X.shape[1]
                theta0 = np.r_[1e-2 * ones(dim)]
                thetaL = np.r_[1e-3 * ones(dim)]
                thetaU = np.r_[20 * ones(dim)]

                models = [GPC(theta0=theta0, thetaL=thetaL, thetaU=thetaU) \
                    for k in range(n_folds)]

    def predict():
        pass



# ----------------------- Loading the data set from R ------------------------
r['load']('./data/modelling.RData')

df = pandas2ri.ri2py(r['data'])
X = df.iloc[:, 0:-1].as_matrix()
y = array(df['is_error'], dtype='int')

n_sample, n_feature = X.shape
n_error = sum(y)

print 'total records: {}'.format(n_sample)
print 'no defect records: {}'.format(n_error)
print 'defect records: {}'.format(n_sample - n_error)


# -------------------------------- Modellling ---------------------------------
# setup saving paths
data_path = './data'
fig_path = './figure/modelling'

try:
    os.makedirs(fig_path)
except:
    pass

is_parallel = False


#model_list = ['svm']

select_defect = ['F4', 'F15', 'F22']

n_defect = len(select_defect)

n_folds = 10
n_rep = 10
n_model = len(model_list)

# evaluation measures
auc_score = zeros((n_defect, n_model, n_folds, n_rep))
acc_rate = zeros((n_defect, n_model, n_folds, n_rep))
mse_score = zeros((n_defect, n_model, n_folds, n_rep))


# perform the mode cross-validation
for p, defect in enumerate(select_defect):

    print
    print 'defect type: {}'.format(defect)
    print

    rate = {}

    # prepare the training set
    idx = [x in [defect, 'OK'] for x in df['Art']]
    data = df.iloc[idx, :]
    X = data.iloc[:, 0:-1].drop('Art', axis=1).as_matrix()
    y = array(data['is_error'], dtype='int')

    for i, model_type in enumerate(model_list):

        for j in range(n_rep):

            print 'model: {}, Repetition: {}'.format(model_type, j+1)

            skf = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
            index = [(train_index, test_index) for train_index, test_index in skf]

            t_start = time.time()


            if is_parallel:

                training_set = [(X[train_index, :], y[train_index]) \
                    for train_index, test_index in index]
                test_set = [(X[test_index, :], y[test_index]) \
                    for train_index, test_index in index]

                # Spawning processes to test kriging mixture
                comm = MPI.COMM_SELF.Spawn(sys.executable, args=['fitting.py'],
                                           maxprocs=n_folds)

                # scatter the models and data
                comm.scatter(models, root=MPI.ROOT)
                comm.scatter([(k, training_set[k], test_set[k]) \
                    for k in range(n_folds)], root=MPI.ROOT)

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
                print 'takes {0:.3f}mins'.format((t_stop - t_start)/60)

            else:   # sequential k-fold cross-validation
                results = []
                for k, idx in enumerate(index):

                    print 'Perform {} fold'.format(k+1)

                    model = models[k]
                    train_index, test_index = idx

                    training_set = X[train_index, :], y[train_index]
                    test_set, target = X[test_index, :], y[test_index]

                    pdb.set_trace()
                    if isinstance(model, SVC_tune):  # tune SVM parameters
                        model.tune(training_set[0], training_set[1],
                                   test_set, target)
                    else:
                        model.fit(*training_set)

                    # Model predictions
                    test_pred = model.predict(test_set)
                    test_probs = model.predict_proba(test_set)

                    # Compute the performance measures
                    mse = mean((test_pred - target) ** 2.0)
                    acc = mean(test_pred == target)
                    _auc = roc_auc_score(target, test_probs[:, 1], 'weighted')
                    fpr, tpr, thresholds = roc_curve(target, test_probs[:, 1])

                    results.append({
                        'index': k,
                        'model': model,
                        'mse': mse,
                        'acc': acc,
                        'auc': _auc,
                        'fpr': fpr,
                        'tpr': tpr
                        })

            # register the measures
            scores = DataFrame([[d['index'], d['acc'], d['mse'], d['auc']] \
                for d in results], columns=['index', 'acc', 'mse', 'auc'])
            scores.sort('index', inplace=True)

            acc_rate[p, i, :, j] = scores['acc']
            auc_score[p, i, :, j] = scores['auc']
            mse_score[p, i, :, j] = scores['mse']

            # record the fpr, tpr rates
            rate[j] = {d['index']: c_[d['fpr'], d['tpr']] for d in results}

        print 'accuracy: {}'.format(mean(acc_rate[p, i, :, :]))
        print 'auc score: {}'.format(mean(auc_score[p, i, :, :]))
        print 'mse: {}'.format(mean(mse_score[p, i, :, :]))

        # ROC plot for each model after CV
        fig, axes = plt.subplots(2, 5, figsize=(fig_width, fig_height), dpi=100)
        axes.shape = (n_rep, )

        for k, ax in enumerate(axes):
            ax.grid(True)
            ax.hold(True)

            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            for n in range(n_folds):
                fpr = rate[k][n][:, 0]
                tpr = rate[k][n][:, 1]
                roc_auc = auc_score[p, i, n, j]

                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0

                ax.plot(fpr, tpr, lw=1, color = _color[n],
                        label='ROC fold %d (area = %0.2f)' % (n+1, roc_auc))

            mean_tpr /= n_folds
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)

            ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6),
                    label='random gusess')
            ax.plot(mean_fpr, mean_tpr, 'k--',
                    label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.set_title('Rep {}'.format(k+1))
            ax.legend(loc="lower right", prop={'size': 10})

        fig.text(0.5, 0.04, 'False Positive Rate', ha='center', va='center',
                 fontsize=17)
        fig.text(0.1, 0.5, 'True Positive Rate', ha='center',
                 va='center', rotation='vertical', fontsize=17)
        fig.suptitle("ROC of model {}".format(model))
        fig.savefig(fig_path + '/ROC-{}-{}.pdf'.format(defect, model))

# Save the data
f = file(data_path + '/cv_comparison.dat', 'w')
cp.dump({'model_list': model_list,
         'n_folds': n_folds,
         'n_reptition': n_rep,
         'acc_rate': acc_rate,
         'mse_rate': mse_score,
         'auc_score': auc_score}, f)


# data plotting
fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))

acc_data = [acc_rate[i, :, :].flatten() for i in range(n_model)]
auc_data = [auc_score[i, :, :].flatten() for i in range(n_model)]
mse_data = [mse_score[i, :, :].flatten() for i in range(n_model)]

axes[0].boxplot(acc_data)
axes[1].boxplot(mse_data)
axes[2].boxplot(auc_data)

axes[0].set_title('accuracy')
axes[1].set_title('MSE')
axes[2].set_title('AUC')

for ax in axes:
    ax.grid(True)
    ax.hold(True)
    ax.set_xticklabels(model_list)

fig.savefig(fig_path + '/cv-comparison.pdf')
