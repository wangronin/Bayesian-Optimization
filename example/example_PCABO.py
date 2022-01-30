import sys

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from bayes_optim import RandomForest, BO, GaussianProcess

sys.path.insert(0, "./")

from bayes_optim.extension import PCABO, RealSpace, KernelPCABO
from bayes_optim.mylogging import eprintf
import benchmark.bbobbenchmarks as bn

np.random.seed(0)
dim = 2
lb, ub = -5, 5
OBJECTIVE_FUNCTION = bn.F17()

def fitness(x):
    # x = np.asarray(x)
    # return np.sum((np.arange(1, dim + 1) * x) ** 2)
    # eprintf("Evaluated solution:", x, "type", type(x))
    if type(x) is np.ndarray:
        x = x.tolist()
    return OBJECTIVE_FUNCTION(np.array(x)) 


space = RealSpace([lb, ub]) * dim
eprintf("new call to PCABO")
opt = PCABO(
    search_space=space,
    obj_fun=fitness,
    DoE_size=5,
    max_FEs=40,
    verbose=True,
    n_point=1,
    n_components=1,
    acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
)

#class _BO(BO):
#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)
#        self._hist_EI = np.zeros(3)
#
#    def ask(self, n_point=None):
#        X = super().ask(n_point=n_point)
#        if self.model.is_fitted:
#            _criter = self._create_acquisition(fun="EI", par={}, return_dx=False)
#            self._hist_EI[(self.iter_count - 1) % 3] = np.mean([_criter(x) for x in X])
#        return X
#
#    def check_stop(self):
#        _delta = self._fBest_DoE - self.fopt
#        if self.iter_count > 1 and np.mean(self._hist_EI[0 : min(3, self.iter_count - 1)]) < 0.01 * _delta:
#            self.stop_dict["low-EI"] = np.mean(self._hist_EI)
#
#        if self.eval_count >= (self.max_FEs / 2):
#            self.stop_dict["max_FEs"] = self.eval_count
#
#        return super().check_stop()
#
#
#search_space = RealSpace([lb, ub]) * dim
#model = GaussianProcess(
#    domain=search_space,
#    n_obj=1,
#    n_restarts_optimizer=dim,
#)
#opt = BO(
#    search_space=space,
#    obj_fun=fitness,
#    model=model,
#    DoE_size=5,
#    max_FEs=50,
#    verbose=True,
#    n_point=1,
#    acquisition_optimization={"optimizer": "OnePlusOne_Cholesky_CMA"},
#    data_file='tmp1.log'
#)

print(opt.run())
