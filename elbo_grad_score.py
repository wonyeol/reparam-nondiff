import numpy as np
import scipy.stats
import autograd
import autograd.scipy as ascipy

import util
from expr import *


#############
# auxiliary #
#############

def norm_logpdf_tht(x, tht):
    return ascipy.stats.norm.logpdf(x, tht[0], util.softplus_anp(tht[1]))

def grad_norm_logpdf_tht(x, tht):
    return util.grad_arg2(norm_logpdf_tht)(x, tht)


########
# init #
########

def init(e): pass


########
# eval #
########

def eval(e, thts, env={}):
    """
    Args:
      - e     : Expr
      - thts  : float array
      - env   : (str -> float) dict
    Returns:
      - retvl : float
      - logpq : float
      - glogq : float list
      - xs    : float list
    where
      - env[var_str] = return value of Var(var_str)
      - retvl = return value
      - logpq = log p(xs,Y) - log q_thts(xs)
      - glogq = \grad_\THT log q_\THT(xs) |_{\THT=thts}
      - xs    = samples values
    here capital math symbols denote vectors.
    """
    
    if isinstance(e, Cnst):
        retvl = e.c
        logpq = 0.0
        glogq = []
        xs    = []

    elif isinstance(e, Var):
        assert(e.v in env)
        retvl = env[e.v]
        logpq = 0.0
        glogq = []
        xs    = []

    elif isinstance(e, Linear):
        retvl = e.c0 + sum([ci*env[vi] for (ci,vi) in e.cv_l])
        logpq = 0.0
        glogq = []
        xs    = []

    elif isinstance(e, App):
        # recursive calls
        num_args = len(e.args)
        (retvl_sub, logpq_sub, glogq_sub, xs_sub)\
            = zip(*[ eval(e.args[i], thts, env) for i in range(num_args) ])

        # compute: all
        op = App.OP_DICT[num_args][e.op]
        retvl = op(*[retvl_sub[i] for i in range(num_args)])
        logpq = np.sum(logpq_sub)
        glogq = util.flatten_list(glogq_sub)
        xs    = util.flatten_list(   xs_sub)
            
    elif isinstance(e, If):
        # recursive calls
        (retvl_1, logpq_1, glogq_1, xs_1) =  eval(e.e1, thts, env)
        (retvl_r, logpq_r, glogq_r, xs_r) = (eval(e.e2, thts, env) if retvl_1 > 0 else\
                                             eval(e.e3, thts, env))

        # compute: all
        retvl = retvl_r
        logpq = logpq_1 + logpq_r
        if retvl_1 > 0: glogq = glogq_1 + glogq_r + [0.]*get_num_thts(e.e3)
        else:           glogq = glogq_1 + [0.]*get_num_thts(e.e2) + glogq_r
        xs    =    xs_1 +    xs_r
            
    elif isinstance(e, Let):
        # recursive calls
        (retvl_1, logpq_1, glogq_1, xs_1) = eval(e.e1, thts, env)
        env_new = util.copy_add_dict(env, {e.v1.v : retvl_1})
        (retvl_2, logpq_2, glogq_2, xs_2) = eval(e.e2, thts, env_new)
        
        # compute: all
        retvl = retvl_2
        logpq = logpq_1 + logpq_2
        glogq = glogq_1 + glogq_2
        xs    =    xs_1 +    xs_2

    elif isinstance(e, Sample):
        # recursive calls
        (retvl_1, logpq_1, glogq_1, xs_1) = eval(e.e1, thts, env)
        (retvl_2, logpq_2, glogq_2, xs_2) = eval(e.e2, thts, env)

        # compute: x_3
        stind = e.stind['thts']
        x_3 = np.random.normal(thts[stind], util.softplus(thts[stind+1]))  # do sampling

        # compute: log p(x|p_loc,p_scale) - log q(x|q_loc,q_scale)
        (p_loc, p_scale) = (retvl_1, retvl_2)
        (q_loc, q_scale) = (thts[stind], util.softplus(thts[stind+1]))
        logpq_3 = (scipy.stats.norm.logpdf(x_3, p_loc, p_scale) -\
                   scipy.stats.norm.logpdf(x_3, q_loc, q_scale))

        # compute: \grad_\tht log q_\tht(x) |_{x=x_3, \tht=thts[stind:stind+2]}
        glogq_3 = list(grad_norm_logpdf_tht(x_3, thts[stind:stind+2]))

        # compute: all
        retvl = x_3
        logpq = logpq_1 + logpq_2 + logpq_3
        glogq = glogq_1 + glogq_2 + glogq_3
        xs    =    xs_1 +    xs_2 +    [x_3]
        
    elif isinstance(e, Fsample):
        # recursive calls
        (retvl_1, logpq_1, glogq_1, xs_1) = eval(e.e1, thts, env)
        (retvl_2, logpq_2, glogq_2, xs_2) = eval(e.e2, thts, env)

        # compute: all
        retvl = np.random.normal(retvl_1, retvl_2)  # do sampling
        logpq = logpq_1 + logpq_2
        glogq = glogq_1 + glogq_2
        xs    =    xs_1 +    xs_2

    elif isinstance(e, Observe): 
        # recursive calls
        num_args = len(e.args)
        (retvl_sub, logpq_sub, glogq_sub, xs_sub)\
            = zip(*[ eval(e.args[i], thts, env) for i in range(num_args) ])

        # compute: log p(c|p_loc,p_scale)
        dstr_logpdf = Observe.DSTR_DICT[e.dstr]
        logpq_cur = dstr_logpdf(e.c1.c, *[retvl_sub[i] for i in range(num_args)])

        # compute: all
        retvl = e.c1.c
        logpq = np.sum(logpq_sub) + logpq_cur
        glogq = util.flatten_list(glogq_sub)
        xs    = util.flatten_list(   xs_sub)

    else: assert(False)
    
    return (retvl, logpq, glogq, xs)


#############
# elbo_grad #
#############

def elbo_grad(e, thts):
    assert(isinstance(e, Expr))
    np.random.seed()  # to give different random seeds for diffferent processors

    (_, logpq, glogq, _) = eval(e, thts)
    res = logpq * np.array(glogq)

    return res
