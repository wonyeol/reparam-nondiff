from __future__ import print_function

import numpy as np
import autograd
import autograd.numpy as anp
import autograd.scipy as ascipy
import matplotlib     # trick for resolving
matplotlib.use('agg') # 'no module named _tkinter' error
import matplotlib.pyplot as plt
import time, sys
import itertools
import pickle


########
# list #
########

def flatten_list(l):
    # ref: https://stackoverflow.com/a/45323085
    return list(itertools.chain.from_iterable(l))


########
# dict #
########

def copy_add_dict(d, other):
    copy = d.copy()
    copy.update(other)
    return copy


########
# math #
########

def softplus    (x): return  np.log(1+ np.exp(x))
def softplus_anp(x): return anp.log(1+anp.exp(x))
def softplus_inv(x):
    """
    - REF: https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/python/ops/distributions/util.py#L1094
    - NOTE: (2) is numerically more stable than (1).
      x = softplus(y) = log(1+exp(y))
      ==> y = log(exp(x)-1) ... (1)
            = log((exp(x)-1)/exp(x)) + log(exp(x))
            = log(1-exp(-x)) + x ... (2)
    """
    return  np.log(1- np.exp(-x))+x

def expon_logpdf(x, lam): return anp.log(lam) - lam*x
def lognorm_logpdf(x, sigma, mu):
    """REF: https://github.com/scipy/scipy/blob/master/scipy/stats/_continuous_distns.py#L3795"""
    return -(anp.log(x)-mu)**2 / (2*sigma**2) - anp.log(anp.sqrt(2*anp.pi) * sigma * x)
def bernoulli_logpdf(k, p): return anp.log(1-p) if k==0 else anp.log(p)
def binom_logpdf(k, n, p):
    """REF: https://github.com/scipy/scipy/blob/master/scipy/stats/_discrete_distns.py#L49"""
    combiln =  ascipy.special.gammaln(n+1) -\
              (ascipy.special.gammaln(k+1) + ascipy.special.gammaln(n-k+1))
    return combiln + k*anp.log(p) + (n-k)*anp.log(1-p)
def geom_logpdf(k, p): return anp.log(p) + (k-1)*anp.log(1-p)


############
# autograd #
############

def invert_arg_ord(f):
    """
    Args:    f : X*Y->Z
    Returns: g : Y*X->Z s.t. g(y,x)=f(x,y)
    """
    return lambda y,x: f(x,y)

def grad_arg2(f):
    """
    Args:    f : X*Y->\R
    Returns: \lambda x0,y0: \grad_y f(x0,y0)
    """
    f_yx       = invert_arg_ord(f)
    grad_y_f_yx = autograd.grad(f_yx)
    grad_y_f_xy = invert_arg_ord(grad_y_f_yx)
    return grad_y_f_xy

def jacobian_arg2(f):
    """
    Args:    f : X*Y->\R
    Returns: \lambda x0,y0: \grad_y f(x0,y0)
    """
    f_yx       = invert_arg_ord(f)
    jcb_y_f_yx = autograd.jacobian(f_yx)
    jcb_y_f_xy = invert_arg_ord(jcb_y_f_yx)
    return jcb_y_f_xy


#########
# print #
#########

bcksp = '\033[D'


##############
# plot graph #
##############

def plot_graph(data_l, func,
               plot_fname='graph.png', text_fname='graph.file',
               legend_l=None, step=1, verbose=True):
    """
    Args:
      - data_l   : (A*B list) list
      - legend_l : str list
      - func     : B->C
      - plot_fname : string
      - text_fname : string
      - step     : int
    Returns:
      Plot a graph for {(x_{i*step}, y_{i*step})}_{i=0,1,...},
      where x_j = data[j][0], y_j = func(data[j][1])
    """
    if verbose: print('plotting... ', end=''); sys.stdout.flush()
    st_time = time.time()
    plt.switch_backend('agg')

    res_xsys = []
    for data in data_l:
        data_plot = data[0::step]
        xs =                [a[0] for a in data_plot]
        ys = list(map(func, [a[1] for a in data_plot]))
        plt.plot(xs, ys)
        res_xsys = res_xsys + [(xs, ys)]

    if legend_l is not None:
        plt.legend(legend_l)
    plt.savefig(plot_fname)

    with open(text_fname, 'wb') as f:
        res = {'xsys_l' : res_xsys, 'legend_l' : legend_l}
        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    
    ed_time = time.time()
    if verbose: print('took %.2f sec' % (ed_time-st_time))

    
# def get_logd_normal(x, m, s):
#     """
#     Args:
#       - x     : value at which we want to compute the density
#       - m     : mean
#       - s     : standard deviation
#     Returns:
#       - logd     : log density
#     """
#     v = s**2
#     logd = - (anp.log(2 * anp.pi * v) / 2) - ((x-m)**2 / (2 * v))
#     return logd
#
# get_grad_logd_normal = autograd.grad(get_logd_normal)
