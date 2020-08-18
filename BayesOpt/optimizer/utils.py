import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from .cma_es import cma_es
from .mies import mies

def argmax_restart(
    obj_func,
    search_space,
    h=None,
    g=None,
    eval_budget=100,
    n_restart=10,
    wait_iter=3,
    optimizer='BFGS',
    logger=None
    ):
    # lists of the best solutions and acquisition values
    # from each restart
    xopt, fopt = [], []  
    best = -np.inf
    wait_count = 0  

    for iteration in range(n_restart):
        x0 = search_space.sampling(N=1, method='uniform')[0]
        
        # TODO: when the surrogate is GP, implement a GA-BFGS hybrid algorithm
        # TODO: BFGS only works with continuous parameters
        # TODO: add constraint handling for BFGS
        if optimizer == 'BFGS':
            mask = np.nonzero(search_space.C_mask | search_space.O_mask)[0]
            bounds = np.array([search_space.bounds[i] for i in mask]) 

            if not all(isinstance(x0, float)):
                raise ValueError('BFGS is not supported with mixed variable types.')

            func = lambda x: tuple(map(lambda x: -1. * x, obj_func(x)))
            xopt_, fopt_, stop_dict = fmin_l_bfgs_b(
                func, x0, pgtol=1e-8, factr=1e6, 
                bounds=bounds, maxfun=eval_budget
            )

            xopt_ = xopt_.flatten().tolist()
            fopt_ = -np.asscalar(fopt_)
            
            if logger is not None and stop_dict["warnflag"] != 0:
                logger.debug(
                    "L-BFGS-B terminated abnormally with the state: %s"%stop_dict
                )
                            
        elif optimizer == 'MIES':
            opt = mies(
                search_space, 
                obj_func, 
                eq_func=h, 
                ineq_func=g,
                max_eval=eval_budget, 
                minimize=False, 
                verbose=False, 
                eval_type='list'
            )                           
            xopt_, fopt_, stop_dict = opt.optimize()

        if fopt_ > best:
            best = fopt_
            wait_count = 0

            if logger is not None:
                logger.debug(
                    'restart : %d - funcalls : %d - Fopt : %f'%(iteration + 1, 
                    stop_dict['funcalls'], fopt_)
                )
        else:
            wait_count += 1

        eval_budget -= stop_dict['funcalls']
        xopt.append(xopt_)
        fopt.append(fopt_)
        
        if eval_budget <= 0 or wait_count >= wait_iter:
            break

    # maximization: sort the optima in descending order
    idx = np.argsort(fopt)[::-1]
    return xopt[idx[0]], fopt[idx[0]]