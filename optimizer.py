from __future__ import print_function

import numpy as np
import pathos.multiprocessing as mp
import time, sys

import util


########
# adam #
########

def adam(grad_func, thts, iter_n,
         lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
         sample_n_grad=1, sample_n_var=0,
         verbose=True, misc=None):
    """
    Args:
      - grad_func : float array -> float array
      - thts      : float array
      - iter_n    : int
    Returns: (int * (float array) * float * float) list
      - (t, tht_t, var_componet(tht_t), var_norm(tht_t))
    Reference:
      - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/adam.py
      - https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
      - pseudo-code:
          m_0 <- 0 (Initialize initial 1st moment vector)
          v_0 <- 0 (Initialize initial 2nd moment vector)
          t <- 0 (Initialize timestep)        
          while (tht_t not converged):
            t <- t + 1
            lr_t <- lr * sqrt(1 - beta2^t) / (1 - beta1^t)
            m_t <- beta1 * m_{t-1} + (1 - beta1) * g
            v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
            tht_t <- tht_{t-1} - lr_t * m_t / (sqrt(v_t) + epsilon)
    """
    if sample_n_var == 0: sample_n_var = sample_n_grad
    sample_n = max(sample_n_grad, sample_n_var)
    
    # init: pool
    n_jobs = min(sample_n, mp.cpu_count())
    if n_jobs > 1: pool = mp.ProcessingPool(n_jobs)

    # print
    if verbose: n_bcksp = 3
    else:       n_bcksp = 0
    print('optimizing using %d processors... ' % n_jobs + ' '*n_bcksp, end='')
    sys.stdout.flush()
    st_time = time.time()

    # main
    ddof = 0 if sample_n_var == 1 else 1
    m = np.zeros(len(thts))
    v = np.zeros(len(thts))
    t = 0
    res = []
    if misc is not None: res_misc = []

    for i in range(iter_n):
        # print
        gap_float = iter_n / 100.
        gap_int = 1 if iter_n < 100 else int(gap_float)
        if i % gap_int == 0:
            if verbose:
                print(util.bcksp*3 + '%02d%%' % int(i/gap_float), end='');
                # print('%02d%%: %r' % (int(i/gap_float), list(thts)))
            sys.stdout.flush()

        # compute: gs
        if n_jobs == 1: gs = [grad_func(thts) for i in range(sample_n)]
        else:           gs = pool.map(grad_func, [thts for i in range(sample_n)])
        if misc is not None:
            repar = [repar for (_,repar,_) in gs]
            corrc = [corrc for (_,_,corrc) in gs]
            gs    = [g     for (g,_,_)     in gs]

        # compute: t, lr_t, g
        t = t+1
        lr_t = lr * np.sqrt(1-np.power(beta2,t)) / (1-np.power(beta1,t))
        g = np.mean(gs[0:sample_n_grad], axis=0)

        # m = beta1 * m + (1-beta1) * g
        # v = beta2 * v + (1-beta2) * g * g
        # thts = thts + lr_t * m / (sqrt(v) + epsilon)
        m = beta1 * m + (1-beta1) *  g
        v = beta2 * v + (1-beta2) * (g**2)
        thts = thts + lr_t * m / (np.sqrt(v) + epsilon)

        # update res
        gs_var_cmp = np.mean(np.var(gs[0:sample_n_var], axis=0, ddof=ddof))
        gs_var_nrm = np.var([np.linalg.norm(_g, 2) for _g in gs[0:sample_n_var]], ddof=ddof)
        res = res + [(t, thts, gs_var_cmp, gs_var_nrm, gs)]
        if misc is not None:
            repar_var_cmp = np.mean(np.var(repar[0:sample_n_var], axis=0, ddof=ddof))
            repar_var_nrm = np.var([np.linalg.norm(_g, 2) for _g in repar[0:sample_n_var]], ddof=ddof)
            corrc_var_cmp = np.mean(np.var(corrc[0:sample_n_var], axis=0, ddof=ddof))
            corrc_var_nrm = np.var([np.linalg.norm(_g, 2) for _g in corrc[0:sample_n_var]], ddof=ddof)
            res_misc = res_misc + [(repar_var_cmp, repar_var_nrm,
                                    corrc_var_cmp, corrc_var_nrm)]

    # print
    ed_time = time.time()
    print(util.bcksp*n_bcksp + ' took %.2f sec' % (ed_time-st_time), end='')

    # variance
    var_cmp_mean = np.mean([gs_var_cmp for (_,_,gs_var_cmp,_,_) in res])
    var_nrm_mean = np.mean([gs_var_nrm for (_,_,_,gs_var_nrm,_) in res])
    print(' [var_cmp=%g, var_nrm=%g]' % (var_cmp_mean, var_nrm_mean), end='')
    if misc is not None:
        repar_var_cmp_mean = np.mean([repar_var_cmp for (repar_var_cmp,_,_,_) in res_misc])
        repar_var_nrm_mean = np.mean([repar_var_nrm for (_,repar_var_nrm,_,_) in res_misc])
        corrc_var_cmp_mean = np.mean([corrc_var_cmp for (_,_,corrc_var_cmp,_) in res_misc])
        corrc_var_nrm_mean = np.mean([corrc_var_nrm for (_,_,_,corrc_var_nrm) in res_misc])
        print(' [repar_var_{cmp,nrm}=(%.5e, %.5e),'
              '  corrc_var_{cmp,nrm}=(%.5e, %.5e)]' %\
              (repar_var_cmp_mean, repar_var_nrm_mean,
               corrc_var_cmp_mean, corrc_var_nrm_mean,), end='')
    
    print('')
    return res
