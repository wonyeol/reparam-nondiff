import numpy as np
import scipy.stats

import util
from expr import *


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
      - xs    : float list
    where
      - env[var_str] = return value of Var(var_str)
      - retvl = return value
      - logpq = log p(X,Y) - log q_\THT(X) |_{X=xs}
      - xs    = sampled values
    here capital math symbols denote vectors.
    """
    
    if isinstance(e, Cnst):
        retvl = e.c
        logpq = 0.0
        xs    = []

    elif isinstance(e, Var):
        assert(e.v in env)
        retvl = env[e.v]
        logpq = 0.0
        xs    = []

    elif isinstance(e, Linear):
        retvl = e.c0 + sum([ci*env[vi] for (ci,vi) in e.cv_l])
        logpq = 0.0
        xs    = []

    elif isinstance(e, App):
        # recursive calls
        num_args = len(e.args)
        (retvl_sub, logpq_sub, xs_sub)\
            = zip(*[ eval(e.args[i], thts, env) for i in range(num_args) ])

        # compute: all
        op = App.OP_DICT[num_args][e.op]
        retvl = op(*[retvl_sub[i] for i in range(num_args)])
        logpq = np.sum(logpq_sub)
        xs    = util.flatten_list(xs_sub)
            
    elif isinstance(e, If):
        # recursive calls
        (retvl_1, logpq_1, xs_1) =  eval(e.e1, thts, env)
        (retvl_r, logpq_r, xs_r) = (eval(e.e2, thts, env) if retvl_1 > 0 else\
                                    eval(e.e3, thts, env))

        # compute: all
        retvl = retvl_r
        logpq = logpq_1 + logpq_r
        xs    =    xs_1 +    xs_r
            
    elif isinstance(e, Let):
        # recursive calls
        (retvl_1, logpq_1, xs_1) = eval(e.e1, thts, env)
        env_new = util.copy_add_dict(env, {e.v1.v : retvl_1})
        (retvl_2, logpq_2, xs_2) = eval(e.e2, thts, env_new)
        
        # compute: all
        retvl = retvl_2
        logpq = logpq_1 + logpq_2
        xs    =    xs_1 +    xs_2

    elif isinstance(e, Sample):
        # recursive calls
        (retvl_1, logpq_1, xs_1) = eval(e.e1, thts, env)
        (retvl_2, logpq_2, xs_2) = eval(e.e2, thts, env)

        # compute: x_3
        stind = e.stind['thts']
        x_3 = np.random.normal(thts[stind], util.softplus(thts[stind+1]))  # do sampling

        # compute: log p(x|p_loc,p_scale) - log q(x|q_loc,q_scale)
        (p_loc, p_scale) = (retvl_1, retvl_2)
        (q_loc, q_scale) = (thts[stind], util.softplus(thts[stind+1]))
        logpq_3 = (scipy.stats.norm.logpdf(x_3, p_loc, p_scale) -\
                   scipy.stats.norm.logpdf(x_3, q_loc, q_scale))

        # compute: all
        retvl = x_3
        logpq = logpq_1 + logpq_2 + logpq_3
        xs    =    xs_1 +    xs_2 +    [x_3]
        
    elif isinstance(e, Fsample):
        # recursive calls
        (retvl_1, logpq_1, xs_1) = eval(e.e1, thts, env)
        (retvl_2, logpq_2, xs_2) = eval(e.e2, thts, env)

        # compute: all
        retvl = np.random.normal(retvl_1, retvl_2)  # do sampling
        logpq = logpq_1 + logpq_2
        xs    =    xs_1 +    xs_2
        
    elif isinstance(e, Observe):
        # recursive calls
        num_args = len(e.args)
        (retvl_sub, logpq_sub, xs_sub)\
            = zip(*[ eval(e.args[i], thts, env) for i in range(num_args) ])

        # compute: log p(c|p_loc,p_scale)
        dstr_logpdf = Observe.DSTR_DICT[e.dstr]
        logpq_cur = dstr_logpdf(e.c1.c, *[retvl_sub[i] for i in range(num_args)])

        # compute: all
        retvl = e.c1.c
        logpq = np.sum(logpq_sub) + logpq_cur
        xs    = util.flatten_list(xs_sub)

    else: assert(False)
    
    return (retvl, logpq, xs)


############
# elbo_val #
############

def elbo_val(e, thts, sample_n=1):
    assert(isinstance(e, Expr))

    sum_logpq = 0.
    for i in range(sample_n):
        (_, logpq, _) = eval(e, thts)
        sum_logpq += logpq

    return sum_logpq / sample_n
