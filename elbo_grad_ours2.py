import logging, sys
import numpy as np
import scipy.stats
import autograd
import autograd.numpy as anp
import autograd.scipy as ascipy
from autograd.builtins import isinstance, list, dict, tuple  # TODO: reconsider to use this line

import util
from expr import *


#############
# auxiliary #
#############

def merge_fs(fs, r0, g):
    """
    Args:    
      - fs : (a -> c) list 
      - r0 : c
      - g  : c * c -> c
    Returns: 
      - f  : a -> c
    where
      f(x) = fold_left g r0 [fs[0](x); fs[1](x); ...; fs[n-1](x)]
    """
    n = len(fs)
    def _f(_x, n=n, fs=fs, r0=r0, g=g):
        res = r0
        for i in range(n):
            res = g(res, fs[i](_x))
        return res
    return _f

def merge_logpqs(logpqs):
    """
    Args:    logpqs : logpq list
    Returns: lambda _thts: \sum_i logpqs[i](_thts)
    """
    def g(z0,z1): return z0+z1
    return merge_fs(logpqs, 0, g)


##############
# eval_repar #
##############
    
def eval_repar(e, thts, env={}):
    """
    Summary: computes the reparameterization term in our estimator.
    Args:
      - e     : Expr
      - thts  : float array
      - env   : (str -> (func * float)) dict
    Returns:
      - ret   : func
      - retvl : float
      - epss  : float array
      - xs    : float array
      - logpq : func
    where
      - env[var_str] = return value of Var(var_str) as (function of \THT, float)
      - ret(\THT) = return value of e (as a function of \THT)
      - retvl  = ret(thts)
      - epss   = values sampled from N(0,1)
      - xs     = T_thts(epss)
      - logpq(\THT) = log p(X,Y) - log q_\THT(X) |_{X=T_\THT(epss)}
    here capital math symbols denote vectors.
    """
    
    if isinstance(e, Cnst):
        ret   = lambda _thts, c=e.c: c
        retvl = ret([])
        epss  = anp.array([])
        xs    = anp.array([])
        logpq = lambda _thts: 0.0

    elif isinstance(e, Var):
        assert(e.v in env)
        (ret, retvl) = env[e.v]
        epss  = anp.array([])
        xs    = anp.array([])
        logpq = lambda _thts: 0.0

    elif isinstance(e, Linear):
        ret   = None # ASSUME: (Linear ...) appear only in the conditional part of If.
        retvl = e.c0 + sum([ci*env[vi][1] for (ci,vi) in e.cv_l])
        epss  = anp.array([])
        xs    = anp.array([])
        logpq = lambda _thts: 0.0
        
    elif isinstance(e, App):
        # recursive calls
        num_args = len(e.args)
        (ret_sub, retvl_sub, epss_sub, xs_sub, logpq_sub)\
            = zip(*[ eval_repar(e.args[i], thts, env) for i in range(num_args) ])

        # compute: all but ret, retvl
        epss  = anp.concatenate( epss_sub)
        xs    = anp.concatenate(   xs_sub)
        logpq = merge_logpqs   (logpq_sub)
        
        # compute: ret, retvl
        op = App.OP_DICT[num_args][e.op]
        ret   = lambda _thts, op=op, ret_sub=ret_sub, num_args=num_args:\
                op(*[  ret_sub[i](_thts) for i in range(num_args)])
        retvl = op(*[retvl_sub[i]        for i in range(num_args)])
            
    elif isinstance(e, If):
        # recursive calls
        (_, retvl_1, epss_1, xs_1, logpq_1)\
            =  eval_repar(e.e1, thts, env)
        (ret_r, retvl_r, epss_r, xs_r, logpq_r)\
            = (eval_repar(e.e2, thts, env) if retvl_1 > 0 else\
               eval_repar(e.e3, thts, env))

        # compute: all
        ret   = ret_r
        retvl = retvl_r
        epss  = anp.concatenate(( epss_1, epss_r ))
        xs    = anp.concatenate((   xs_1,   xs_r ))
        logpq = merge_logpqs   ([logpq_1, logpq_r])
            
    elif isinstance(e, Let):
        # recursive calls
        (ret_1, retvl_1, epss_1, xs_1, logpq_1) = eval_repar(e.e1, thts, env)
        env_new = util.copy_add_dict(env, {e.v1.v : (ret_1, retvl_1)})
        (ret_2, retvl_2, epss_2, xs_2, logpq_2) = eval_repar(e.e2, thts, env_new)
        
        # compute: all
        ret   = ret_2
        retvl = retvl_2
        epss  = anp.concatenate(( epss_1, epss_2 ))
        xs    = anp.concatenate((   xs_1,   xs_2 ))
        logpq = merge_logpqs   ([logpq_1, logpq_2])

    elif isinstance(e, Sample):
        # recursive calls
        (ret_1, retvl_1, epss_1, xs_1, logpq_1) = eval_repar(e.e1, thts, env)
        (ret_2, retvl_2, epss_2, xs_2, logpq_2) = eval_repar(e.e2, thts, env)

        # compute: all but logpq
        stind = e.stind['thts']
        eps_3     = np.random.normal(0, 1) # do sampling
        eps2x_cur = lambda _tht, eps=eps_3: _tht[0] + util.softplus_anp(_tht[1]) * eps
        eps2x_3   = lambda _thts, eps2x_cur=eps2x_cur, stind=stind: eps2x_cur(_thts[stind:stind+2])
        x_3       = eps2x_3(thts)

        ret   = lambda _thts, eps2x_3=eps2x_3: eps2x_3(_thts)
        retvl = x_3  # use current thts value to compute return value
        epss  = anp.concatenate(( epss_1, epss_2, anp.array([eps_3]) ))
        xs    = anp.concatenate((   xs_1,   xs_2, anp.array([  x_3]) ))

        # compute: logpq
        def logpq_3(_thts, ret=ret, ret_1=ret_1, ret_2=ret_2, stind=stind):
            # compute: log p(x|p_loc,p_scale) - log q(x|q_loc,q_scale)
            x       = ret  (_thts)
            p_loc   = ret_1(_thts)
            p_scale = ret_2(_thts)
            q_loc   =                   _thts[stind]
            q_scale = util.softplus_anp(_thts[stind+1])
            return (ascipy.stats.norm.logpdf(x, p_loc, p_scale) -\
                    ascipy.stats.norm.logpdf(x, q_loc, q_scale))
        
        logpq = merge_logpqs([logpq_1, logpq_2, logpq_3])

    elif isinstance(e, Fsample):
        # recursive calls
        (ret_1, retvl_1, epss_1, xs_1, logpq_1) = eval_repar(e.e1, thts, env)
        (ret_2, retvl_2, epss_2, xs_2, logpq_2) = eval_repar(e.e2, thts, env)

        # compute: all
        x_3 = np.random.normal(retvl_1, retvl_2)  # do sampling
        ret   = lambda _thts, x_3=x_3: x_3
        retvl = x_3
        epss  = anp.concatenate(( epss_1, epss_2 ))
        xs    = anp.concatenate((   xs_1,   xs_2 ))
        logpq = merge_logpqs   ([logpq_1, logpq_2])

    elif isinstance(e, Observe): 
        # recursive calls
        num_args = len(e.args)
        (ret_sub, retvl_sub, epss_sub, xs_sub, logpq_sub)\
            = zip(*[ eval_repar(e.args[i], thts, env) for i in range(num_args) ])

        # compute: all but logpq
        ret   = lambda _thts, c=e.c1.c: c
        retvl = ret([])
        epss  = anp.concatenate( epss_sub)
        xs    = anp.concatenate(   xs_sub)

        # compute: logpq
        dstr_logpdf = Observe.DSTR_DICT[e.dstr]
        def logpq_cur(_thts, dstr_logpdf=dstr_logpdf, c=e.c1.c,
                      ret_sub=ret_sub, num_args=num_args):
            # compute: log p(c|p_loc,p_scale)
            return dstr_logpdf(c, *[ret_sub[i](_thts) for i in range(num_args)])

        logpq = merge_logpqs(list(logpq_sub) + [logpq_cur])

    else: assert(False)
    return (ret, retvl, epss, xs, logpq)


################
# eval_surface #
################
    
def eval_surface(e, thts, xs, if_ind, if_tf, env={}):
    """
    Summary: computes a part of the correction term in our estimator.
    Args:
      - e      : Expr
      - xs     : ({'sample','fsample'} -> float array) dict
      - if_ind : int
      - if_tf  : bool
      - env      : (str -> float) dict
    Returns:
      - retvl    : float
      - logpq    : float
    where
      - if_ind = ind of If expr that we are now focusing
      - if_tf  = which branch to take when we encounter If expr of if_ind
      - env[var_str] = return value of Var(var_str) as float
      - retvl  = return value
      - logpq = log p(xs,Y) - log q_thts(xs), by following if_tf on if_ind
    """
    
    if isinstance(e, Cnst):
        retvl = e.c
        logpq = 0.0

    elif isinstance(e, Var):
        assert(e.v in env)
        retvl = env[e.v]
        logpq = 0.0

    elif isinstance(e, Linear):
        retvl = e.c0 + sum([ci*env[vi] for (ci,vi) in e.cv_l])
        logpq = 0.0
        
    elif isinstance(e, App):
        # recursive calls
        num_args = len(e.args)
        (retvl_sub, logpq_sub)\
            = zip(*[ eval_surface(e.args[i], thts, xs, if_ind, if_tf, env)
                     for i in range(num_args) ])
        
        # compute: all
        op = App.OP_DICT[num_args][e.op]
        retvl = op(*[retvl_sub[i] for i in range(num_args)])
        logpq = np.sum(logpq_sub)
            
    elif isinstance(e, If):
        # recursive calls
        (retvl_1, logpq_1) = eval_surface(e.e1, thts, xs, if_ind, if_tf, env)

        if e.ind == if_ind:
            e_next = e.e2 if if_tf == True else\
                     e.e3
        else:
            e_next = e.e2 if retvl_1 > 0 else\
                     e.e3
        (retvl_r, logpq_r) = eval_surface(e_next, thts, xs, if_ind, if_tf, env)

        # compute: all
        retvl = retvl_r
        logpq = logpq_1 + logpq_r
            
    elif isinstance(e, Let):
        # recursive calls
        (retvl_1, logpq_1) = eval_surface(e.e1, thts, xs, if_ind, if_tf, env)
        env_new = util.copy_add_dict(env, {e.v1.v : retvl_1})
        (retvl_2, logpq_2) = eval_surface(e.e2, thts, xs, if_ind, if_tf, env_new)
        
        # compute: all
        retvl = retvl_2
        logpq = logpq_1 + logpq_2

    elif isinstance(e, Sample):
        # recursive calls
        (retvl_1, logpq_1) = eval_surface(e.e1, thts, xs, if_ind, if_tf, env)
        (retvl_2, logpq_2) = eval_surface(e.e2, thts, xs, if_ind, if_tf, env)

        # load: x_3
        x_3 = xs['sample'][e.ind]

        # compute: log p(x|p_loc,p_scale) - log q(x|q_loc,q_scale)
        stind = e.stind['thts']
        (p_loc, p_scale) = (retvl_1, retvl_2)
        (q_loc, q_scale) = (thts[stind], util.softplus(thts[stind+1]))
        logpq_3 = (scipy.stats.norm.logpdf(x_3, p_loc, p_scale) -\
                   scipy.stats.norm.logpdf(x_3, q_loc, q_scale))

        # compute: all
        retvl = x_3
        logpq = logpq_1 + logpq_2 + logpq_3        

    elif isinstance(e, Fsample): 
        # recursive calls
        (retvl_1, logpq_1) = eval_surface(e.e1, thts, xs, if_ind, if_tf, env)
        (retvl_2, logpq_2) = eval_surface(e.e2, thts, xs, if_ind, if_tf, env)

        # load: x_3
        x_3 = xs['fsample'][e.ind]

        # compute: all
        retvl = x_3
        logpq = logpq_1 + logpq_2

    elif isinstance(e, Observe): 
        # recursive calls
        num_args = len(e.args)
        (retvl_sub, logpq_sub)\
            = zip(*[ eval_surface(e.args[i], thts, xs, if_ind, if_tf, env)
                     for i in range(num_args) ])
        
        # compute: log p(c|p_loc,p_scale)
        dstr_logpdf = Observe.DSTR_DICT[e.dstr]
        logpq_cur = dstr_logpdf(e.c1.c, *[retvl_sub[i] for i in range(num_args)])
        
        # compute: all
        retvl = e.c1.c
        logpq = np.sum(logpq_sub) + logpq_cur

    else: assert(False)
    return (retvl, logpq)


#############
# do_sample #
#############

def do_sample(e, thts, env={}):
    """
    Summary: do sampling for Sample and Fsample
    Args:
      - e     : Expr
      - thts  : float array
      - env   : (str -> float) dict
    Returns:
      - retvl : float
      - xs_s  : float list
      - xs_f  : float list
    where
      - env[var_str] = return value of Var(var_str) as float
      - retvl = return value
      - xs_s  = sampled values for Sample  (from approximating distribution)
      - xs_f  = sampled values for Fsample (from prior distribution)
    """
    
    if isinstance(e, Cnst):
        retvl = e.c
        xs_s = []
        xs_f = []

    elif isinstance(e, Var):
        assert(e.v in env)
        retvl = env[e.v]
        xs_s = []
        xs_f = []

    elif isinstance(e, Linear):
        retvl = e.c0 + sum([ci*env[vi] for (ci,vi) in e.cv_l])
        xs_s = []
        xs_f = []

    elif isinstance(e, App):
        # recursive calls
        num_args = len(e.args)
        (retvl_sub, xs_s_sub, xs_f_sub)\
            = zip(*[ do_sample(e.args[i], thts, env) for i in range(num_args) ])

        # compute: all
        op = App.OP_DICT[num_args][e.op]
        retvl = op(*[retvl_sub[i] for i in range(num_args)])
        xs_s = util.flatten_list(xs_s_sub)
        xs_f = util.flatten_list(xs_f_sub)

    elif isinstance(e, If):
        # recursive calls
        (retvl_1, xs_s_1, xs_f_1) = do_sample(e.e1, thts, env)
        e_next = e.e2 if retvl_1 > 0 else\
                 e.e3
        (retvl_r, xs_s_r, xs_f_r) = do_sample(e_next, thts, env)
        
        # compute: all
        retvl = retvl_r
        xs_s = xs_s_1 + xs_s_r
        xs_f = xs_f_1 + xs_f_r
        
    elif isinstance(e, Let):
        # recursive calls
        (retvl_1, xs_s_1, xs_f_1) = do_sample(e.e1, thts, env)
        env_new = util.copy_add_dict(env, {e.v1.v : retvl_1})
        (retvl_2, xs_s_2, xs_f_2) = do_sample(e.e2, thts, env_new)
        
        # compute: all
        retvl = retvl_2
        xs_s = xs_s_1 + xs_s_2
        xs_f = xs_f_1 + xs_f_2
        
    elif isinstance(e, Sample):
        # recursive calls
        (retvl_1, xs_s_1, xs_f_1) = do_sample(e.e1, thts, env)
        (retvl_2, xs_s_2, xs_f_2) = do_sample(e.e2, thts, env)
        
        # sample: x_3 from approximating distribution
        stind = e.stind['thts']
        x_3 = np.random.normal(thts[stind], util.softplus(thts[stind+1]))
        
        # compute: all
        retvl = x_3
        xs_s = xs_s_1 + xs_s_2 + [x_3]  # add to xs_s
        xs_f = xs_f_1 + xs_f_2
        
    elif isinstance(e, Fsample):
        # recursive calls
        (retvl_1, xs_s_1, xs_f_1) = do_sample(e.e1, thts, env)
        (retvl_2, xs_s_2, xs_f_2) = do_sample(e.e2, thts, env)
        
        # sample: x_3 from prior distribution
        x_3 = np.random.normal(retvl_1, retvl_2)
        
        # compute: all
        retvl = x_3
        xs_s = xs_s_1 + xs_s_2
        xs_f = xs_f_1 + xs_f_2 + [x_3]  # add to xs_f
        
    elif isinstance(e, Observe):
        # recursive calls
        num_args = len(e.args)
        (retvl_sub, xs_s_sub, xs_f_sub)\
            = zip(*[ do_sample(e.args[i], thts, env) for i in range(num_args) ])

        # compute: all
        retvl = e.c1.c
        xs_s = util.flatten_list(xs_s_sub)
        xs_f = util.flatten_list(xs_f_sub)
        
    else: assert(False)
    return (retvl, xs_s, xs_f)
            

########
# init #
########

x2eps = lambda x, thts: (x - thts[0])/util.softplus_anp(thts[1])
grad_tht_x2eps = util.grad_arg2(x2eps)

tot_cnt     = {}
tot_var2ind = {}
tot_ind2e   = {}

def init(e):
    global tot_cnt, tot_var2ind, tot_ind2e
    tot_cnt     = decorate_ind(e)
    tot_var2ind = get_var2ind (e)
    tot_ind2e   = get_ind2e   (e)

def  decorate_ind(e): return _decorate_ind(e, {'sample':0, 'fsample':0, 'if':0})
def _decorate_ind(e, cnt):
    if   isinstance(e, Cnst):   pass
    elif isinstance(e, Var):    pass
    elif isinstance(e, Linear): pass

    elif isinstance(e, App):
        for ei in e.args:
            cnt = _decorate_ind(ei, cnt)

    elif isinstance(e, If):
        cnt_prev = dict(cnt)
        # record cnt of If
        e.ind = cnt['if']; cnt['if'] = cnt['if']+1
        cnt = _decorate_ind(e.e1, cnt)
        cnt = _decorate_ind(e.e2, cnt)
        cnt = _decorate_ind(e.e3, cnt)
        # ASSUME: no Sample and Fsample inside If's
        assert(cnt_prev['sample' ] == cnt['sample' ] and
               cnt_prev['fsample'] == cnt['fsample'])
            
    elif isinstance(e, Let):
        cnt = _decorate_ind(e.v1, cnt)
        cnt = _decorate_ind(e.e1, cnt)
        cnt = _decorate_ind(e.e2, cnt)

    elif isinstance(e, Sample):
        # record cnt of Sample
        e.ind = cnt['sample']; cnt['sample'] = cnt['sample']+1 
        cnt = _decorate_ind(e.e1, cnt)
        cnt = _decorate_ind(e.e2, cnt)

    elif isinstance(e, Fsample):
        # record cnt of Fsample
        e.ind = cnt['fsample']; cnt['fsample'] = cnt['fsample']+1
        cnt = _decorate_ind(e.e1, cnt)
        cnt = _decorate_ind(e.e2, cnt)

    elif isinstance(e, Observe): 
        for ei in e.args:
            cnt = _decorate_ind(ei, cnt)
        cnt = _decorate_ind(e.c1, cnt)

    else: assert(False)
    return cnt

def  get_var2ind(e): return _get_var2ind(e, {})
def _get_var2ind(e, res):
    if   isinstance(e, Cnst):   pass
    elif isinstance(e, Var):    pass
    elif isinstance(e, Linear): pass

    elif isinstance(e, App):
        for ei in e.args:
            res = _get_var2ind(ei, res)

    elif isinstance(e, If):
        res = _get_var2ind(e.e1, res)
        res = _get_var2ind(e.e2, res)
        res = _get_var2ind(e.e3, res)
            
    elif isinstance(e, Let):
        # add to res_dict
        if   isinstance(e.e1, Sample ): res[e.v1.v] = ('sample' , e.e1.ind)  
        elif isinstance(e.e1, Fsample): res[e.v1.v] = ('fsample', e.e1.ind)
        res = _get_var2ind(e.e1, res)
        res = _get_var2ind(e.e2, res)

    elif isinstance(e, Sample):
        res = _get_var2ind(e.e1, res)
        res = _get_var2ind(e.e2, res)

    elif isinstance(e, Fsample):
        res = _get_var2ind(e.e1, res)
        res = _get_var2ind(e.e2, res)

    elif isinstance(e, Observe): 
        for ei in e.args:
            res = _get_var2ind(ei, res)
        res = _get_var2ind(e.c1, res)

    else: assert(False)
    return res

def  get_ind2e(e): return _get_ind2e(e, {})
def _get_ind2e(e, res):
    if   isinstance(e, Cnst):   pass
    elif isinstance(e, Var):    pass
    elif isinstance(e, Linear): pass

    elif isinstance(e, App):
        for ei in e.args:
            res = _get_ind2e(ei, res)

    elif isinstance(e, If):
        # add to res_dict
        res[('if', e.ind)] = e
        res = _get_ind2e(e.e1, res)
        res = _get_ind2e(e.e2, res)
        res = _get_ind2e(e.e3, res)
            
    elif isinstance(e, Let):
        res = _get_ind2e(e.e1, res)
        res = _get_ind2e(e.e2, res)

    elif isinstance(e, Sample):
        res = _get_ind2e(e.e1, res)
        res = _get_ind2e(e.e2, res)

    elif isinstance(e, Fsample):
        # add to res_dict
        res[('fsample', e.ind)] = e
        res = _get_ind2e(e.e1, res)
        res = _get_ind2e(e.e2, res)

    elif isinstance(e, Observe): 
        for ei in e.args:
            res = _get_ind2e(ei, res)
        res = _get_ind2e(e.c1, res)

    else: assert(False)
    return res

    
#############
# elbo_grad #
#############

def elbo_grad(e, thts, misc=None):
    assert(isinstance(e, Expr))
    np.random.seed()  # to give different random seeds for diffferent processors

    # compute: reparam term
    (_, _, _, _, logpq_fun) = eval_repar(e, thts)
    reparam_term = autograd.grad(logpq_fun)(thts)

    # compute: correction term
    num_sample  = tot_cnt['sample']
    num_fsample = tot_cnt['fsample']
    correctn_term = np.zeros(2*num_sample)  # num_thts = 2*num_sample

    # compute: if_ind_l
    subsample_n = 1  # -1:no subsampling, >0:do subsampling
    if subsample_n == -1: if_ind_l = range(tot_cnt['if']); subsample_n = tot_cnt['if']
    else:                 if_ind_l = np.random.randint(0, tot_cnt['if'], subsample_n)
        
    for if_ind in if_ind_l:
        e_cond = tot_ind2e[('if', if_ind)].e1
        assert(isinstance(e_cond, Linear))

        # compute: coeffs, nz_ind
        coeffs = {'sample'  : np.zeros(num_sample),
                  'fsample' : np.zeros(num_fsample)}
        nz_ind = -1
        for (ci, vi) in e_cond.cv_l:
            # ASSUME: vi's are distinct
            (s_or_f, ind) = tot_var2ind[vi]
            coeffs[s_or_f][ind] = ci
            if s_or_f == 'sample' and ci != 0.:
                nz_ind = ind  # ind of Sample whose coeff is nonzero in e_cond
        if nz_ind == -1: continue  # correction term for current e_cond is 0

        # compute: xs
        if num_fsample == 0:
            xs_sample  = [np.random.normal(thts[2*i], util.softplus(thts[2*i+1]))
                          for i in range(num_sample)]
            xs_fsample = []
            # xs_fsample = [np.random.normal(tot_ind2e[('fsample',i)].e1.c,
            #                                tot_ind2e[('fsample',i)].e2.c)
            #               for i in range(num_fsample)]
        else:
            (_, xs_sample, xs_fsample) = do_sample(e, thts)
        xs = {'sample'  : xs_sample,
              'fsample' : xs_fsample}
        xs['sample'][nz_ind] = 0.
        xs['sample'][nz_ind] = -(e_cond.c0
                                 + np.inner(coeffs['sample' ], xs['sample' ])
                                 + np.inner(coeffs['fsample'], xs['fsample'])) \
                                 / coeffs['sample'][nz_ind]

        # compute: q_tht_xn, n_tht
        q_tht_xn = scipy.stats.norm.pdf(xs['sample'][nz_ind],
                                        thts[2*nz_ind], util.softplus(thts[2*nz_ind+1]))
        sgn_cn = np.sign(coeffs['sample'][nz_ind])
        n_tht = [-sgn_cn * util.softplus(thts[2*i+1]) * (coeffs['sample'][i] / coeffs['sample'][nz_ind])
                 for i in range(num_sample)] # for true branch

        # compute: f_tht_{tt,ff}
        (_, f_tht_tt) = eval_surface(e, thts, xs, if_ind, True)
        (_, f_tht_ff) = eval_surface(e, thts, xs, if_ind, False)

        # update: correctn_term
        if f_tht_tt == f_tht_ff: continue
        for tht_ind in range(2*num_sample):
            # compute: vdotn
            x_ind = tht_ind/2
            v_x_ind = grad_tht_x2eps(xs['sample'][nz_ind], thts[2*x_ind:2*x_ind+2])[tht_ind%2]
            vdotn = v_x_ind * n_tht[x_ind] # for true branch

            # update
            grad_tht_ind = (f_tht_tt - f_tht_ff) * q_tht_xn * vdotn
            correctn_term[tht_ind] += grad_tht_ind

    # return
    correctn_term *= tot_cnt['if'] / np.float64(subsample_n)  # consider subsampling
    res = reparam_term + correctn_term
    if misc is None: return  res
    else:            return (res, reparam_term, correctn_term)
