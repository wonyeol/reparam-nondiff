import autograd

from expr import *
import util, elbo_grad_ours2 # elbo_grad_ours


########
# init #
########

def init(e): pass


#############
# elbo_grad #
#############

def elbo_grad(e, thts):
    assert(isinstance(e, Expr))

    (_, _, _, _, logpq_fun) = elbo_grad_ours2.eval_repar(e, thts)
    reparam_term = autograd.grad(logpq_fun)(thts)
    
    # (_, _, epss, _, _, _, _, logpq_fun) = elbo_grad_ours.eval(e, thts)
    # gtht_logpq = util.jacobian_arg2(logpq_fun)(epss, thts)
    # res = gtht_logpq

    return reparam_term
