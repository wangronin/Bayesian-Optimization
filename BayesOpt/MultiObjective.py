# TODO: implement those!
class SMS_BO(BO):
    pass


class MOBO_D(BO):
    """Decomposition-based Multi-Objective Bayesian Optimization (MO-EGO/D) 
    """
    # TODO: this number should be set according to the capability of the server
    # TODO: implement Tchebycheff scalarization
    __max_procs__ = 16  # maximal number of processes

    def _eval(x, _eval_type, obj_func, _space=None, logger=None, runs=1, pickling=False):
        """evaluate one solution

        Parameters
        ----------
        x : bytes,
            serialization of the Solution instance
        """
        if pickling:
            # TODO: move the pickling/unpickling operation to class 'Solution'
            x = dill.loads(x)
        fitness_, n_eval = x.fitness.flatten(), x.n_eval

        if hasattr(obj_func, '__call__'):   # vector-valued obj_func
            if _eval_type == 'list':
                ans = [obj_func(x.tolist()) for i in range(runs)]
            elif _eval_type == 'dict':
                ans = [obj_func(_space.to_dict(x)) for i in range(runs)]

            # TODO: this should be done per objective fct.
            fitness = np.sum(np.asarray(ans), axis=0)

            # TODO: fix it
            # x.fitness = fitness / runs if any(np.isnan(fitness_)) \
            #     else (fitness_ * n_eval + fitness) / (x.n_eval + runs)
            x.fitness = fitness / runs 

        elif hasattr(obj_func, '__iter__'):  # a list of  obj_func
            for i, obj_func in enumerate(obj_func):
                try:
                    if _eval_type == 'list':
                        ans = [obj_func(x.tolist()) for i in range(runs)]
                    elif _eval_type == 'dict':
                        ans = [obj_func(_space.to_dict(x)) for i in range(runs)]
                except Exception as ex:
                    logger.error('Error in function evaluation: {}'.format(ex))
                    return

                fitness = np.sum(ans)
                x.fitness[0, i] = fitness / runs if np.isnan(fitness_[i]) \
                    else (fitness_[i] * n_eval + fitness) / (x.n_eval + runs)

        x.n_eval += runs
        return dill.dumps(x)
    
    def __init__(self, n_obj=2, aggregation='WS', n_point=5, n_job=1, *argv, **kwargs):
        """
        Arguments
        ---------
        n_point : int,
            the number of evaluated points in each iteration
        aggregation: str or callable,
            the scalarization method/function. Supported options are:
                'WS' : weighted sum
                'Tchebycheff' : Tchebycheff scalarization
        """
        super(MOBO_D, self).__init__(*argv, **kwargs)
        self.n_point = int(n_point)
        # TODO: perhaps leave this an input parameter
        self.mu = 2 * self.n_point   # the number of generated points
        self.n_obj = int(n_obj)
        assert self.n_obj > 1

        if isinstance(self.minimize, bool):
            self.minimize = [self.minimize] * self.n_obj
        elif hasattr(self.minimize, '__iter__'):
            assert len(self.minimize) == self.n_obj

        self.minimize = np.asarray(self.minimize)

        if hasattr(self.obj_func, '__iter__'):
            assert self.n_obj == len(self.obj_func)
        
        assert self.n_obj == len(self.surrogate)
        self.n_job = min(MOBO_D.__max_procs__, self.mu, n_job)

        # TODO: implement the Tchebycheff approach
        if isinstance(aggregation, str):
            assert aggregation in ['WS', 'Tchebycheff']
        else:
            assert hasattr(aggregation, '__call__')
        self.aggregation = aggregation

        # generate weights
        self.weights = np.random.rand(self.mu, self.n_obj)
        self.weights /= np.sum(self.weights, axis=1).reshape(self.mu, 1)
        self.labels_ = KMeans(n_clusters=self.n_point).fit(self.weights).labels_
        self.frange = np.zeros(self.n_obj)

    def evaluate(self, data, runs=1):
        """Evaluate the candidate points and update evaluation info in the dataframe
        """
        _eval_fun = functools.partial(MOBO_D._eval, _eval_type=self._eval_type, _space=self._space,
                                      obj_func=self.obj_func, logger=self.logger, runs=runs,
                                      pickling=self.n_job > 1)
        if len(data.shape) == 1:
            _eval_fun(data)
        else: 
            if self.n_job > 1:
                 # parallel execution using multiprocessing
                if self._parallel_backend == 'multiprocessing':
                    data_pickle = [dill.dumps(d) for d in data]
                    # __ = self.pool.map(_eval_fun, data_pickle)
                    __ = Parallel(n_jobs=self.n_job)(delayed(_eval_fun)(d) for d in data_pickle)

                    x = [dill.loads(_) for _ in __]
                    self.eval_count += runs * len(data)
                    for i, k in enumerate(data):
                        data[i].fitness = x[i].fitness
                        data[i].n_eval = x[i].n_eval
            else:
                for x in data:
                    _eval_fun(x)

    def fit_and_assess(self):
        def _fit(surrogate, X, y):
            surrogate.fit(X, y)
            y_hat = surrogate.predict(X)
            r2 = r2_score(y, y_hat)    
            return surrogate, r2
        
        # NOTE: convert the fitness to minimization problem
        # objective values that are subject to maximization is revert to mimization 
        self.y = self.data.fitness.copy()
        self.y *= np.asarray([-1] * self.data.N).reshape(-1, 1) ** (~self.minimize)

        ymin, ymax = np.min(self.y), np.max(self.y)
        if np.isclose(ymin, ymax): 
            raise Exception('flat objective value!')

        # self._y: normalized objective values
        self._y = (self.y - ymin) / (ymax - ymin)

        # fit the surrogate models
        if self.n_job > 1 and 11 < 2:  
            data = zip(*[(self.surrogate[i], self.data, self._y[:, i]) for i in range(self.n_obj)])
            __ = Parallel(n_jobs=self.n_job)(delayed(_fit)(d) for d in data)
            # __ = self.pool.map(_fit, *data)
        else:               
            __ = []
            for i in range(self.n_obj):
                __.append(list(_fit(self.surrogate[i], self.data, self._y[:, i])))

        self.surrogate, r2 = tuple(zip(*__))
        for i in range(self.n_obj):
            self.logger.info('F{} Surrogate model r2: {}'.format(i + 1, r2[i]))

    def step(self):
        # self._initialize()  
        
        X = self.select_candidate() 
        self.evaluate(X, runs=self.init_n_eval)
        self.data += X
        
        self.fit_and_assess()     
        self.iter_count += 1

        # TODO: implement a faster algorithm to detect non-dominated point only!
        # non-dominated sorting: self.y takes minimization issue into account
        nd_idx = non_dominated_set_2d(self.y)

        # xopt is the set of the non-dominated point now
        self.xopt = self.data[nd_idx]  
        self.logger.info('{}iteration {}, {} points in the Pareto front: {}\n{}'.format(bcolors.WARNING, 
            self.iter_count, len(self.xopt), bcolors.ENDC, str(self.xopt)))

        if self.data_file is not None:
            self.xopt.to_csv(self.data_file)

        return self.xopt, self.xopt.fitness
    
    def select_candidate(self):
        _ = self.arg_max_acquisition()
        X, value = np.asarray(_[0], dtype='object'), np.asarray(_[1])
        
        X_ = []
        # select the best point from each cluster
        # NOTE: "the best point" means the maximum of infill criteria 
        for i in range(self.n_point):
            v = value[self.labels_ == i]
            idx = np.nonzero(v == np.max(v))[0][0]
            X_.append(X[self.labels_ == i][idx].tolist())

        X = Solution(X_, index=len(self.data) + np.arange(len(X_)), 
                     var_name=self.var_names, n_obj=self.n_obj)
        X = self._remove_duplicate(X)

        # if the number of new design sites obtained is less than required,
        # draw the remaining ones randomly
        if len(X) < self.n_point:
            self.logger.warn("iteration {}: duplicated solution found " 
                             "by optimization! New points is taken from random "
                             "design".format(self.iter_count))
            N = self.n_point - len(X)
            s = self._space.sampling(N, method='uniform')
            X = Solution(X.tolist() + s, index=len(self.data) + np.arange(self.n_point),
                         var_name=self.var_names, n_obj=self.n_obj)

        return X

    def _acquisition(self, surrogate=None, plugin=None, dx=False):
        """Generate Infill Criteria based on surrogate models

        Parameters
        ----------
        surrogate : class instance
            trained surrogate model
        plugin : float,
            the minimal objective value used in improvement-based infill criteria
            Note that it should be given in the original scale
        """
        # objective values are normalized
        if plugin is None:
            plugin = 0

        # NOTE: the following 'minimize' parameter is set to always 'True'
        # as 
        if self.infill == 'EI':
            acquisition_func = EI(surrogate, plugin, minimize=True)
        elif self.infill == 'PI':
            acquisition_func = PI(surrogate, plugin, minimize=True)
        elif self.infill == 'MGFI':
            acquisition_func = MGFI(surrogate, plugin, minimize=True, t=self.t)
        elif self.infill == 'UCB':
            raise NotImplementedError
                
        return functools.partial(acquisition_func, dx=dx) 

    # TODO: implement evolutionary algorithms, e.g., MOEA/D to optimize of all subproblems simultaneously
    def arg_max_acquisition(self, plugin=None):
        """Global Optimization of the acqusition function / Infill criterion
        Arguments
        ---------
        plugin : float,
            the cut-off value for improvement-based criteria
            it is set to the current minimal target value

        Returns
        -------
            candidates: tuple of list,
                candidate solution (in list)
            values: tuple of float,
                criterion value of the candidate solution
        """
        self.logger.debug('infill criteria optimziation...')
        
        dx = True if self._optimizer == 'BFGS' else False
        surrogates = (SurrogateAggregation(self.surrogate, weights=w) for w in self.weights)
        gmin = [np.min(self._y.dot(w)) for w in self.weights]
        criteria = (self._acquisition(s, gmin[i], dx=dx) for i, s in enumerate(surrogates))

        if self.n_job > 1:
            __ = Parallel(n_jobs=self.n_job)(delayed(self._argmax_multistart)(_) \
                for _ in criteria)
            # __ = self.pool.map(self._argmax_multistart, [_ for _ in criteria])
        else:
            __ = [list(self._argmax_multistart(_)) for _ in criteria]

        candidates, values = tuple(zip(*__))
        return candidates, values