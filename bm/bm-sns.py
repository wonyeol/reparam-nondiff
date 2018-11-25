# this file is based on bm4-4-b.py.

"""
SNS Message Example from Bayesian Method for Hackers.
We use log-normal instead of exponential.
"""

import numpy as np
import scipy.stats
import autograd.numpy as anp
from expr import *

inv_cdf = [scipy.stats.norm.ppf(i/75.) for i in range(76)]
obs = [None,
       13., 24.,  8., 24.,  7., 35., 14., 11., 15., 11.,
       22., 22., 11., 57., 11., 19., 29.,  6., 19., 12.,
       22., 12., 18., 72., 32.,  9.,  7., 13., 19., 23.,
       27., 20.,  6., 17., 13., 10., 14.,  6., 16., 15.,
        7.,  2., 15., 15., 19., 70., 49.,  7., 53., 22.,
       21., 31., 19., 11., 18., 20., 12., 35., 17., 23.,
       17.,  4.,  2., 31., 30., 13., 27.,  0., 39., 37.,
        5., 14., 13., 22.]
obs_mean = np.mean(obs[1:])
lognorm_std  = np.sqrt(np.log(2.))                  #= 0.8326
lognorm_mean = np.log(obs_mean) - lognorm_std**2/2  #= 2.6362
# ==> mean(m0) = obs_mean, var(m0) = obs_mean**2

def e0(e_last):
    res =\
        Let(Var('x0'), Sample(Cnst(lognorm_mean), Cnst(lognorm_std)),
        Let(Var('x1'), Sample(Cnst(lognorm_mean), Cnst(lognorm_std)),
        Let(Var('m0'), App('exp', [Var('x0')]),
        Let(Var('m1'), App('exp', [Var('x1')]),
        Let(Var('z'),  Sample(Cnst(0.), Cnst(1.)),
        e_last )))))
    return res

def ei(i, e_last):
    obs_cur = 'obs%d' % i
    res =\
        Let(Var(obs_cur), If(Linear(-inv_cdf[i], [(1., 'z')]),
                             Observe('poisson', [Var('m0')], Cnst(obs[i])),
                             Observe('poisson', [Var('m1')], Cnst(obs[i]))),
        e_last)
    return res

en = Cnst(0.)
e = unroll_loop(e0, ei, en, range(2, 75, 2))  # range(1, 75, 1)
thts_init = anp.array(
    [lognorm_mean, util.softplus_inv(lognorm_std)] * 2 +
    [0.,           util.softplus_inv(1.)]
)

optz_cfg = {'iter_n'  : 10000,
            'lr'      : 0.001,
            'sample_n_grad': 1,
            'sample_n_var' : 0}
plot_cfg = {'sample_n': 100,
            'step'    : 100}
compare = {'ours2' : 'elbo_grad_ours2.py',
           'score' : 'elbo_grad_score.py',
           'repar' : 'elbo_grad_repar.py'}

"""
* Anglican results

- 1st run:
mean(x0): 3.2247944
mean(x1): 2.937255
mean(z-as-day): 25.44857

- 2nd run:
mean(x0): 3.223499
mean(x1): 2.9379528
mean(z-as-day): 25.48667
"""

"""
* NOTE

def lognorm_info(mean, std):
  print('mean = %g' % np.exp(mean + (std**2)/2))
  print('mode = %g' % np.exp(mean - std**2))
  print('variance = %g' % ((np.exp(std**2)-1) * np.exp(2*mean + std**2)))

lognorm_info(0.8415, 0.7151)
lognorm_info(2.6362, 0.8326)
"""
