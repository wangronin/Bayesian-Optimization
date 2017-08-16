# -*- coding: utf-8 -*-

# Author: Hao Wang <wangronin@gmail.com>
#         Bas van Stein <bas9112@gmail.com>


import numpy as np
from numpy import array, ones, inner, dot, diag, size
from numpy.random import shuffle
from copy import deepcopy, copy

from multiprocessing import Pool


from sklearn.utils.validation import check_is_fitted

import itertools    
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import skfuzzy as fuzz
from .gprhao import GaussianProcess_extra
#from .regressiontree import IncrementalRegressionTree


def train_model( clus, model, training_set):
    shape = np.array(training_set[0]).shape
    while True:  
        try:
            model = model.fit(*training_set)
            break
        except ValueError:
            #print('Current nugget setting is too small!' +\
            #        ' It will be tuned up automatically')
            model.nugget *= 10
    return model

def train_modelstar(inputdata):
    return train_model(*inputdata)


MACHINE_EPSILON = np.finfo(np.double).eps
class OWCK(GaussianProcess_extra):
    """The Optimal Weighted Cluster Kriging/Gaussian Process class
    
    This class inherited from GaussianProcess class in sklearn library
    Most of the parameters are contained in sklearn.gaussian_process.
    
    Please check the docstring of Gaussian Process parameters in sklearn.
    Only newly introduced parameters are documented below.

    Parameters
    ----------
    n_cluster : int, optional
        The number of clusters, determines the number of the Gaussian Process
        model to build. It is the speed-up factor in OWCK.
    min_leaf_size : int, optional
        if min_leaf_size > 0, min_leaf_size is used to determine the number of clusters for
        the model tree clustering method.
    cluster_method : string, optional
        The clustering algorithm used to partition the data set.
        Built-in clustering algorithm are:
            'k-mean', 'GMM', 'fuzzy-c-mean', 'random', 'tree'
            Note that GMM, fuzzy-c-mean  are fuzzy clustering algorithms 
            With these algorithms you can set the overlap you desire.
            Tree is a non-fuzzy algorithm using local models per leaf in a regression tree
            The tree algorithm is also able to update the model with new records
    overlap : float, optional
        The percentage of overlap when using a fuzzy cluster method.
        Each cluster will be of the same size.
    is_parallel : boolean, optional
        A boolean switching parallel model fitting on. If it is True, then
        all the underlying Gaussian Process model will be fitted in parallel,
        supported by MPI. Otherwise, all the models will be fitted sequentially.
        
    Attributes
    ----------
    cluster_label : the cluster label of the training set after clustering
    clusterer : the clustering algorithm used.
    models : a list of (fitted) Gaussian Process models built on each cluster.
    
    References
    ----------
    
    .. [SWKBE15] `Bas van Stein, Hao Wang, Wojtek Kowalczyk, Thomas Baeck 
        and Michael Emmerich. Optimally Weighted Cluster Kriging for Big 
        Data Regression. In 14th International Symposium, IDA 2015, pages 
        310-321, 2015`
        http://link.springer.com/chapter/10.1007%2F978-3-319-24465-5_27#
    """

    def __init__(self, regr='constant', corr='squared_exponential', 
                 n_cluster=8, min_leaf_size=0, cluster_method='k-mean', overlap=0.0, beta0=None, 
                 storage_mode='full', verbose=False, theta0=0.1, thetaL=None, 
                 thetaU=None, sigma2=None, optimizer='BFGS', random_start=1, 
                 normalize=False, nugget=10. * MACHINE_EPSILON, random_state=None, 
                 nugget_estim=True, is_parallel=False):
        
        super(OWCK, self).__init__(regr=regr, corr=corr, 
                 beta0=beta0, verbose=verbose, 
                 theta0=theta0, thetaL=thetaL, thetaU=thetaU, sigma2=sigma2,
                 optimizer=optimizer, random_start=random_start, 
                 normalize=normalize, nugget=nugget, nugget_estim=nugget_estim,
                 random_state=random_state)

        self.empty_model = GaussianProcess_extra(regr=regr, corr=corr, 
                 beta0=beta0, verbose=verbose, 
                 theta0=theta0, thetaL=thetaL, thetaU=thetaU, sigma2=sigma2,
                 optimizer=optimizer, random_start=random_start, normalize=normalize,
                 nugget=nugget, nugget_estim=nugget_estim, random_state=random_state)
        
        self.n_cluster = n_cluster
        self.is_parallel = is_parallel
        self.verbose = verbose
        self.overlap = overlap #overlap for fuzzy clusters
        self.min_leaf_size = min_leaf_size
        self.regr_label = regr
        self.fitted = False
        
        if cluster_method not in ['k-mean', 'GMM', 'fuzzy-c-mean', 'random', 'tree']:
            raise Exception('{} clustering is not supported!'.format(cluster_method))
        else:
            self.cluster_method = cluster_method
            
    
    def __clustering(self, X, y=None):
        """
        The clustering procedure of the Optimal Weighted Clustering Gaussian 
        Process. This function should not be called externally
        """

        self.sizeX = len(X)
        
        if self.cluster_method == 'k-mean':
            clusterer = KMeans(n_clusters=self.n_cluster)
            clusterer.fit(X)
            self.cluster_label = clusterer.labels_
            self.clusterer = clusterer
        elif self.cluster_method == 'tree':
            
            # if (self.min_leaf_size > 0):
            #     self.minsamples = self.min_leaf_size
            #     tree = IncrementalRegressionTree(min_samples_leaf=self.min_leaf_size)
            # else:
            #     self.minsamples = int(len(X)/(self.n_cluster))
            #     tree = IncrementalRegressionTree(min_samples_leaf=self.minsamples)
            
            # tree.fit(X,y)
            # labels = tree.apply(X)
            # clusters = np.unique(labels)
            # k = len(clusters)
            # if self.verbose:
            #     print ("leafs:",k)
            # self.n_cluster = k
            # self.leaf_labels = np.unique(labels)
            # self.cluster_label = labels
            # self.clusterer = tree
            pass
            
        elif self.cluster_method == 'random':
            r = self.n_sample % self.n_cluster
            m = (self.n_sample - r) / self.n_cluster
            self.cluster_label = array(range(self.n_cluster) * m + range(r))
            self.clusterer = None
            shuffle(self.cluster_label)
        elif self.cluster_method == 'GMM':    #GMM from sklearn
            self.clusterer = GaussianMixture(n_components=self.n_cluster, n_init=10)
            self.clusterer.fit(X)
            self.cluster_labels_proba = self.clusterer.predict_proba(X)
            self.cluster_label = self.clusterer.predict(X)
        elif self.cluster_method == 'fuzzy-c-mean': #Fuzzy C-means from sklearn
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, self.n_cluster, 2, error=0.000005, maxiter=10000, init=None)
            self.clusterer = cntr #save the centers for cmeans_predict
            self.cluster_labels_proba = u.T
            self.cluster_labels_proba = np.array(self.cluster_labels_proba)
            self.cluster_label = np.argmax(u, axis=0)
            self.cluster_label = np.array(self.cluster_label)
        
    def __fit(self, X, y):
        """
        The Optimal Weighted Cluster Gaussian Process model fitting method.
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
        ocwk : self
            A fitted Cluster Gaussian Process model object awaiting data to 
            perform predictions.
        """
        self.n_sample, self.n_feature = X.shape
        
        if y.shape[0] != self.n_sample:
            raise Exception('Training input and target do not match!')
        
        # clustering
        self.__clustering(X,y)
        
        # model creation
        self.models = None;
        if (self.cluster_method == 'tree'):
            self.models = [deepcopy(self) for i in self.leaf_labels]
        else:
            self.models = [deepcopy(self) for i in range(self.n_cluster)]
        
        for m in self.models:
            m.__class__ = GaussianProcess_extra
        
        self.X = X
        self.y = y
        
        # model fitting
        if self.is_parallel:     # parallel model fitting
            #now using parralel threading
            #
            # prepare the training set for each GP model    
                
            if (self.cluster_method=='k-mean' or self.cluster_method=='random'):
                idx = [self.cluster_label == i for i in range(self.n_cluster)]
            elif (self.cluster_method=='tree'):
                idx = [self.cluster_label == self.leaf_labels[i] for i in range(self.n_cluster)]
                if (self.verbose):
                    print "len cluster", len(idx)
            else:
                targetMemberSize = (len(self.X) / self.n_cluster)*(1.0+self.overlap)
                idx = []

                minindex = np.argmin(self.y)
                maxindex = np.argmax(self.y)
                for i in range(self.n_cluster):
                    idx_temp = np.argsort(self.cluster_labels_proba[:,i])[-targetMemberSize:]
                    if (minindex not in idx_temp):
                        idx_temp = np.hstack((idx_temp,[minindex]))
                    if (maxindex not in idx_temp):
                        idx_temp = np.hstack((idx_temp,[maxindex]))
                    idx.append(idx_temp)


            training = [(X[index, :], y[index]) for index in idx]
            training_set = itertools.izip(range(self.n_cluster),deepcopy(self.models),training )
            
            pool = Pool(self.n_cluster) 
            models = pool.map(train_modelstar, training_set)
            pool.close()
            pool.join()
            self.models = models
            
            #print models
            #
            '''
            raise Exception('Parallel mode has been disabled for now.')
            # spawning processes...
            #os.chdir('/home/wangronin')
            comm = MPI.COMM_SELF.Spawn(sys.executable, 
                                       args=['-m', 'owck.OWCK_slave'],
                                       maxprocs=self.n_cluster)
            
            # prepare the training set for each GP model    
                
            if (self.cluster_method=='k-mean' or self.cluster_method=='random'):
                idx = [self.cluster_label == i for i in range(self.n_cluster)]
            elif (self.cluster_method=='tree'):
                idx = [self.cluster_label == self.leaf_labels[i] for i in range(self.n_cluster)]
                if (verbose):
                    print "len cluster",len(idx)
            else:
                targetMemberSize = (len(self.X) / self.n_cluster)*(1.0+self.overlap)
                idx = []

                minindex = np.argmin(self.y)
                maxindex = np.argmax(self.y)
                for i in range(self.n_cluster):
                    idx_temp = np.argsort(self.cluster_labels_proba[:,i])[-targetMemberSize:]
                    if (minindex not in idx_temp):
                        idx_temp = np.hstack((idx_temp,[minindex]))
                    if (maxindex not in idx_temp):
                        idx_temp = np.hstack((idx_temp,[maxindex]))
                    idx.append(idx_temp)


            training_set = [(X[index, :], y[index]) for index in idx]
            
            # scatter the models and data
            comm.scatter([(k, training_set[k]) \
                for k in range(self.n_cluster)], root=MPI.ROOT)
            comm.scatter(self.models, root=MPI.ROOT)
           
            
            # Synchronization while the slave process are performing 
            # heavy computations...
            comm.Barrier()
                
            # Gether the fitted model from the childrenn process
            # Note that 'None' is only valid in master-slave working mode
            results = comm.gather(None, root=MPI.ROOT)
            
            # keep the fitted model align with their cluster
            fitted = DataFrame([[d['index'], d['model']] \
                for d in results], columns=['index', 'model'])
            fitted.sort('index', ascending=[True], inplace=True)
            
            self.models[:] = fitted['model']
                
            # free all slave processes
            comm.Disconnect()
            '''
        
        else:                    # sequential model fitting
            # get min and max value indexes such that no cluster gets 
            # only one value instances.
#            minindex = np.argmin(self.training_y)
#            maxindex = np.argmax(self.training_y)
            for i in range(self.n_cluster):                    
               
                if (self.cluster_method=='k-mean' or self.cluster_method=='random'):
                    idx = self.cluster_label == i
                elif (self.cluster_method=='tree'):
                    idx = self.cluster_label == self.leaf_labels[i]
                else:
                    targetMemberSize = (len(self.X) / self.n_cluster)*(1.0+self.overlap)
                    idx = []

                    minindex = np.argmin(self.y)
                    maxindex = np.argmax(self.y)
                    # TODO: fix line here
                    idx = np.argsort(self.cluster_labels_proba[:,i])[-int(targetMemberSize):]
                    if (minindex not in idx):
                        idx = np.hstack((idx,[minindex]))
                    if (maxindex not in idx):
                        idx = np.hstack((idx,[maxindex]))

                model = self.models[i]
                # TODO: discuss this will introduce overlapping samples
#                idx[minindex] = True
#                idx[maxindex] = True
                
                # dirty fix so that low nugget errors will increase the
                # nugget till the model fits
                while True:  
                    try:
                        # super is needed here to call the 'fit' function in the 
                        # parent class (GaussianProcess_extra)
                        if (self.cluster_method=='tree' and self.verbose):
                            print 'leaf: ', self.leaf_labels[i]
                        length_lb = 1e-10
                        length_ub = 1e2
                        X = self.X[idx, :]
                        x_lb, x_ub = X.min(0), X.max(0)
                        model.thetaL = length_ub ** -2.  / (x_ub - x_lb) ** 2. * np.ones(self.n_feature)
                        model.thetaU = length_lb ** -2.  / (x_ub - x_lb) ** 2. * np.ones(self.n_feature)
                        
                        model.fit(self.X[idx, :], self.y[idx])
                        break
                    except Exception as e:
                        print e
                        if self.verbose:
                            print('Current nugget setting is too small!' +\
                                ' It will be tuned up automatically')
                        #pdb.set_trace()
                        model.noise_var *= 10
    
    def gradient(self, x):
        """
        Calculate the gradient of the posterior mean and variance
        """
        check_is_fitted(self, 'X')
        x = np.atleast_2d(x)
        
        if self.cluster_method == 'tree':
            idx = self.clusterer.apply(x.reshape(1, -1))[0]
            active_GP_idx = np.nonzero(self.leaf_labels == idx)[0][0]
            active_GP = self.models[active_GP_idx]
             
            y_dx, mse_dx = active_GP.gradient(x)
             
        elif self.cluster_method == 'GMM':
            # TODO: implement this 
            pass
        
        elif self.cluster_method in ['random', 'k-mean']:
            par = {}
            _ = self.predict(x, eval_MSE=False, par_out=par)

            weights = par['weights'].reshape(-1, 1)
            y = par['y'].reshape(-1, 1)
            mse =  par['mse'].reshape(-1, 1)
            normalized_mse = par['mse_normalized'].reshape(-1, 1) 
            U = par['U'].reshape(-1, 1)

            y_jac, mse_jac = zip(*[model.gradient(x) for model in self.models])
            y_jac, mse_jac = np.c_[y_jac], np.c_[mse_jac]

            M = (1.  / normalized_mse).sum()
            tmp = np.dot(mse_jac, (1. / normalized_mse ** 2.) / U)
            weights_jacobian =  -mse_jac * normalized_mse.T ** -2. / U.T / M \
                + (np.repeat(tmp, len(weights), axis=1) / normalized_mse.T) / M ** 2.

            y_dx = np.dot(y_jac, weights) + np.dot(weights_jacobian, y)
            mse_dx = np.dot(mse_jac, weights ** 2.) + np.dot(weights.T * weights_jacobian, mse)
    
        return y_dx, mse_dx
    
    
    def __mse_upper_bound(self, model):
        """
        This function computes the tight upper bound of the Mean Square Error(
        Kriging variance) for the underlying Posterior Gaussian Process model, 
        whose usage should be subject to Simple or Ordinary Kriging (constant trend)
        Parameters
        ----------
        model : a fitted Gaussian Process/Kriging model, in which 'self.regr'
                should be 'constant'
        Returns
        ----------
        upper_bound : the upper bound of the Mean Squared Error
        """
        
        if self.regr_label != 'constant':
            raise Exception('MSE upper bound only exists for constant trend')
            
        C = model.C
        if C is None:
        # Light storage mode (need to recompute C, F, Ft and G)
            if model.verbose:
                print("This GaussianProcess used 'light' storage mode "
                          "at instantiation. Need to recompute "
                          "autocorrelation matrix...")
            _, par = model.reduced_likelihood_function()
            model.C = par['C']
            model.Ft = par['Ft']
            model.G = par['G']
    
        n_samples, n_features = model.X.shape
        tmp = 1 / model.G ** 2
    
        upper_bound = np.sum(model.sigma2 * (1 + tmp))
        return upper_bound
        
    def __check_duplicate(self, X, y):
        
        # TODO: show a warning here
        X = np.atleast_2d(X)
        new_X = []
        new_Y = []
        for i, x in enumerate(X):
            idx = np.nonzero(np.all(np.isclose(self.X, x), axis=1))[0]
            
            if len(idx) == 0:
                new_X.append(x)
                new_Y.append(y[i])
            
            if y[i] != self.y[idx]:
                raise Exception('The same input can not have different respones!')
        
        return np.array(new_X), new_Y

    def updateModel(self, newX, newY):
        """
        Deprecated function, just call fit with new database.
        """
        newY = newY.reshape(-1, 1)
        #print newY.shape, self.y.shape
        X = np.r_[self.X, newX]
        y = np.r_[self.y, newY]
        self.fit(X, y)

    def update_data(self, X, y):
        self.X = X
        self.y = y

        # note that the clusters are not rebuilt
        if self.cluster_method == 'tree':
            self.cluster_label = self.clusterer.apply(self.X)

            for i, model in enumerate(self.models):
                idx = self.cluster_label == self.leaf_labels[i]
                if not np.any(idx):
                    raise Exception('No data point in cluster {}!'.format(i+1))
                model.update_data(self.X[idx, :], self.y[idx])
        else:
            # TODO: to implement for the rest options
            pass 

        return self

    def fit(self, newX, newY, re_estimate_all=False):
        """
        Add several instances to the data and rebuild models
        newX is a 2d array of (instances,features) and newY a vector
        """
        if not hasattr(self, 'X'):
            self.__fit(newX, newY)
            return
            
        newX, newY = self.__check_duplicate(newX, newY)
            
        if self.cluster_method == 'tree':

            #first update our data
            if len(newY) != 0:
                self.X = np.r_[self.X, newX]
                self.y = np.r_[self.y, newY]
            #self.X = np.append(self.X, newX, axis=0)
            #self.y = np.append(self.y, newY)

            #check the size of the new data
            if re_estimate_all:
                #in this case build additional models
                if self.verbose:
                    print("refitting all models")
                self.__fit(self.X, self.y)
            elif len(self.X) > (self.sizeX + self.minsamples*2.0):
                #in this case build additional models if needed
                if self.verbose:
                    print("refitting new models")
                    #print("Current tree")
                    #print(self.clusterer)
                rebuildmodels = np.unique(self.clusterer.apply(newX))
                rebuildmodelstemp = []
                rebuild_index = 0;
                self.cluster_label = self.clusterer.apply(self.X)
                new_leaf_labels = []
                for i in rebuildmodels:
                    leafindex = np.where(self.leaf_labels==i)[0][0]

                    idx = self.cluster_label == i
                    #check size of model
                    if (len(idx) > self.minsamples*2.0):
                        if self.verbose:
                            print("Trying to split leaf node",i)
                        #split the leaf and fit 2 additional models
                        new_labels = []
                        if self.clusterer.split_terminal(i,self.X[idx, :], self.y[idx]):
                            self.cluster_label = self.clusterer.apply(self.X)
                            new_labels = np.unique(self.cluster_label)
                            self.n_cluster = len(new_labels)
                        delete_old = False
                        for n in new_labels:
                            if n not in self.leaf_labels:
                                delete_old = True
                                new_leafindex = np.where(new_labels==n)[0][0]
                                if self.verbose:
                                    print("New model with id",new_leafindex)
                                    #print self.leaf_labels
                                new_model = deepcopy(self.empty_model)
                                self.models.append(new_model)
                                self.leaf_labels = np.append(self.leaf_labels,n)
                                #rebuildmodelstemp.append(new_leafindex)
                                new_leaf_labels.append(n)
                        if delete_old:
                            self.leaf_labels = np.delete(self.leaf_labels, leafindex)
                            del(self.models[leafindex])
                            #if self.verbose:
                                #print("New tree")
                                #print(self.clusterer)
                                #print self.leaf_labels
                    else:
                        #just refit this model
                        #rebuildmodelstemp.append(leafindex)
                        new_leaf_labels.append(i)


                for n in new_leaf_labels:
                    rebuildmodelstemp.append(np.where(self.leaf_labels==n)[0][0])


                rebuildmodels = np.unique(np.array(rebuildmodelstemp,dtype=int))
                labels = self.clusterer.apply(self.X)
                self.cluster_label = labels
                self.leaf_labels = np.unique(labels)

                for i in rebuildmodels:
                    
                    idx = self.cluster_label == self.leaf_labels[i]
                    if self.verbose:
                        print("updating model on position "+str(i)+" attached to leaf id "+str(self.leaf_labels[i])+" and "+str(sum(idx))+" data points")
                    model = self.models[i]
                    while True:  
                        try:
                            # super is needed here to call the 'fit' function in the 
                            # parent class (GaussianProcess)
                            
                            model.fit(self.X[idx, :], self.y[idx])
                            break
                        except ValueError:
                            if self.verbose:
                                print('Current nugget setting is too small!' +\
                                    ' It will be tuned up automatically')
                            model.nugget *= 10
            else:
                rebuildmodels = np.unique(self.clusterer.apply(newX))
                rebuildmodelstemp = []
                for i in rebuildmodels:
                    rebuildmodelstemp.append(np.where(self.leaf_labels==i)[0][0])
                rebuildmodels = np.array(rebuildmodelstemp,dtype=int)
                labels = self.clusterer.apply(self.X)
                self.cluster_label = labels
                if self.is_parallel:     # parallel model fitting
                    idx = [self.cluster_label == self.leaf_labels[i] for i in rebuildmodels]
                    modelstosend = [deepcopy(self.models[i]) for i in rebuildmodels]
                    training = [(self.X[index, :], self.y[index]) for index in idx]
                    training_set = itertools.izip(rebuildmodels,modelstosend,training )
                    
                    pool = Pool(self.n_cluster) 
                    models = pool.map(train_modelstar, training_set)
                    pool.close()
                    pool.join()
                    for i in range(len(rebuildmodels)):
                        self.models[rebuildmodels[i]] = models[i]

                    
                else:# is_parralel = false

                    for i in rebuildmodels:
                        if self.verbose:
                            print("updating model "+str(i))
                        idx = self.cluster_label == self.leaf_labels[i]
                        model = self.models[i]
                        while True:  
                            try:
                                # super is needed here to call the 'fit' function in the 
                                # parent class (GaussianProcess)
                                
                                model.fit(self.X[idx, :], self.y[idx])
                                break
                            except ValueError:
                                if self.verbose:
                                    print('Current nugget setting is too small!' +\
                                        ' It will be tuned up automatically')
                                model.nugget *= 10
        else:
            #rebuild all models
            self.X = np.r_[self.X, newX]
            self.y = np.r_[self.y, newY]
            self.__fit(self.X, self.y)
    
    # TODO: implementating batch_size option to reduce the memory usage
    def predict(self, X, eval_MSE=False, par_out=None):
        """
        This function evaluates the Optimal Weighted Gaussian Process model at x.
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
        batch_size : integer, Not available yet 
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
        
        X = np.atleast_2d(X)
        X = X.T if size(X, 1) != self.n_feature else X
    
        n_eval, n_feature = X.shape
        
        if n_feature != self.n_feature:
            raise Exception('Dimensionality does not match!')
        
        if self.cluster_method == 'tree':
            pred = np.zeros(n_eval)
            if eval_MSE:
                mse = np.zeros(n_eval)
            
            for i, x in enumerate(X):
#                modelindex = self.clusterer
                ix = self.clusterer.apply(x.reshape(1, -1))
                model = self.models[np.where(self.leaf_labels == ix)[0][0]]

                _ = model.predict(x.reshape(1, -1), eval_MSE)
                
                if eval_MSE:
                    pred[i], mse[i] = _
                else:
                    pred[i] = _

            if eval_MSE:
                return pred, mse
            else:
                return pred

        elif self.cluster_method in ['random', 'k-mean']:
            # compute predictions and MSE from all underlying GP models
            # super is needed here to call the 'predict' function in the 
            # parent class
        
            res = array([model.predict(X, eval_MSE=True) \
                for model in self.models])
            
            # compute the upper bound of MSE from all underlying GP models
            mse_upper_bound = array([self.__mse_upper_bound(model) \
                for model in self.models])
                    
            if np.any(mse_upper_bound == 0):
                raise Exception('Something weird happened!')
                    
            pred, mse = res[:, 0, :], res[:, 1, :] 
            normalized_mse = mse / mse_upper_bound.reshape(-1, 1)
            
            # inverse of the MSE matrices
            Q_inv = [diag(1.0 / normalized_mse[:, i]) for i in range(n_eval)]
            
            _ones = ones(self.n_cluster)
            weight = lambda Q_inv: dot(_ones, Q_inv)
            normalizer = lambda Q_inv: dot(dot(_ones, Q_inv), _ones.reshape(-1, 1))
            
            # compute the weights of convex combination
            weights = array([weight(q_inv) / normalizer(q_inv) for q_inv in Q_inv])
            # make sure the weights sum to 1...  
            if np.any(abs(np.sum(weights, axis=1) - 1.0) > 1e-8):
                raise Exception('Computed weights do not sum to 1!')
            
            # convex combination of predictions from the underlying GP models
            pred_combined = array([inner(pred[:, i], weights[i, :]) \
                for i in range(n_eval)])

            if par_out is not None:
                par_out['weights'] =  weights
                par_out['y'] =  pred
                par_out['mse'] =  mse
                par_out['mse_normalized'] = normalized_mse
                par_out['U'] = mse / normalized_mse
            
            # if overall MSE is needed        
            if eval_MSE:
                mse_combined = array([inner(mse[:, i], weights[i, :]**2) \
                    for i in range(n_eval)])
            
                return pred_combined, mse_combined
            
            else:
                return pred_combined

        elif self.cluster_method == 'GMM':
            # TODO: implement the MSE calculation for 'GMM' approach: mixed of Gaussian processes
            pass
                
    
if __name__ == '__main__':
    
    X = np.linspace(0, 1, 100).reshape(-1, 1)
    y = np.sin(X)
    
    model = OWCK(regr='constant', corr='absolute_exponential', 
         n_cluster=2, min_leaf_size=0, cluster_method='tree', 
         overlap=0.0, beta0=None,
         storage_mode='full', verbose=True, theta0=np.random.rand(), thetaL=[1e-10], 
         thetaU=[10], optimizer='fmin_cobyla', random_start=1, nugget=0.0001,
         normalize=False, is_parallel=False)
         
    model.fit(X, y)
    
    newX = np.array([[1.2]])
    newY = np.sin(newX)
    
    model.fit(newX, newY)