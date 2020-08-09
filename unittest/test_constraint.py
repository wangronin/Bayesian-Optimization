import numpy as np

from BayesOpt import BO
from BayesOpt.Surrogate import RandomForest
from BayesOpt.SearchSpace import ContinuousSpace, OrdinalSpace, NominalSpace
from BayesOpt.optimizer import mies


def obj_func(x):
    x_r, x_i, x_d = np.array(x[:2]), x[2], x[3]
    if x_d == 'OK':
        tmp = 0
    else:
        tmp = 1
    return np.sum((x_r + np.array([2, 2])) ** 2) + abs(x_i - 10) * 10 + tmp 

def eq_func(x):
    x_r = np.array(x[:2])
    return np.sum(x_r ** 2) - 2

def ineq_func(x):
    x_r = np.array(x[:2])
    return np.sum(x_r) + 1

space = (ContinuousSpace([-10, 10]) * 2) + OrdinalSpace([5, 15]) + \
    NominalSpace(['OK', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])

if 11 < 2:
    opt = mies(space, obj_func, eq_func=eq_func, ineq_func=ineq_func, 
               max_eval=1e3, verbose=True)
    xopt, fopt, stop_dict = opt.optimize()

else:
    model = RandomForest(levels=space.levels)
    opt = BO(space, obj_func, model, eq_fun=None, ineq_fun=None, 
             minimize=True,
             n_init_sample=3, max_eval=50, verbose=True, optimizer='MIES')
    xopt, fopt, stop_dict = opt.run()
        
    # if 11 < 2:  
    #     N = int(50)
    #     max_eval = int(1000)
    #     fopt = np.zeros((1, N))
    #     hist_sigma = np.zeros((100, N))
    #     hist_fitness = np.zeros((100, N))
        
    #     for i in range(N):
    #         opt = mies(space, sphere, max_eval=max_eval, verbose=False)
    #         xopt, fopt[0, i], stop_dict, hist_fitness[:, i], hist_sigma[:, i] = opt.optimize()
        
    #     import matplotlib.pyplot as plt
        
    #     with plt.style.context('ggplot'):
    #         fig0, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 10), dpi=100)
    #         for i in range(N):
    #             ax0.semilogy(range(100), hist_fitness[:, i])
    #             ax1.semilogy(range(100), hist_sigma[:, i])
                
    #         ax0.set_xlabel('iteration')
    #         ax0.set_ylabel('fitness')
    #         ax1.set_xlabel('iteration')
    #         ax1.set_ylabel('Step-sizes')
            
    #         fig0.suptitle('Sphere {}D'.format(dim))
    #         plt.show()