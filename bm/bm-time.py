# this file is based on bm8-2.py

"""
Model for Influenza Mortality Data (Example 6.22 from Shumway and Stoffer's "Time Series Analysis and Its Application".
"""

import autograd.numpy as anp
from expr import *

mortality_data_full = [
        0.2902977, 0.5702096, 0.6384278, 0.3135166, 0.2047724, 0.1885323, 
        0.1798065, 0.1867087, 0.1871117, 0.2157304, 0.2385934, 0.2572363, 
        0.6381312, 0.5195216, 0.2968231, 0.2514277, 0.2144966, 0.1964111, 
        0.2343862, 0.2095384, 0.21776, 0.2450697, 0.2554203, 0.3357194, 
        0.5968794, 0.5156948, 0.3085772, 0.2569596, 0.2194814, 0.2158074, 
        0.2240929, 0.2217431, 0.2388084, 0.2528328, 0.260521, 0.2822165, 
        0.8113721, 0.4458291, 0.3415985, 0.2774243, 0.2484958, 0.2525427, 
        0.2466902, 0.2452006, 0.2279679, 0.2610293, 0.3177998, 0.7298681, 
        0.281731, 0.3030367, 0.2915253, 0.2453148, 0.2051037, 0.1845583, 
        0.2046436, 0.1828399, 0.1960763, 0.2203844, 0.224994, 0.3036654, 
        0.8192756, 0.4376872, 0.3834813, 0.2919304, 0.255642, 0.2369918, 
        0.2495123, 0.2280816, 0.2335083, 0.2679917, 0.3030445, 0.3594536, 
        0.3743171, 0.3746391, 0.3398281, 0.2909505, 0.2403667, 0.2269297, 
        0.2134564, 0.2074597, 0.2099793, 0.2464345, 0.2768291, 0.3359459, 
        0.4932865, 0.5692177, 0.3593959, 0.2741868, 0.2424912, 0.2241473, 
        0.228084, 0.2283657, 0.2282342, 0.2579092, 0.2909701, 0.3113483, 
        0.49372, 0.4728154, 0.3094317, 0.2343683, 0.200722, 0.1906996, 
        0.1877136, 0.1990874, 0.1908602, 0.2111834, 0.2084524, 0.2510464, 
        0.3096678, 0.3330646, 0.349702, 0.3066515, 0.223837, 0.2030033, 
        0.2231299, 0.1891271, 0.1997522, 0.2269953, 0.2295186, 0.3198083, 
        0.5712671, 0.4351815, 0.282585, 0.2381904, 0.2196089, 0.190482, 
        0.1926964, 0.1853681, 0.1943872, 0.2290728, 0.225355, 0.2569408 ]

mortality_data_short = mortality_data_full[:12]
data = mortality_data_short

def e_exp(e_arg):
    return App('exp', [e_arg])

def e_log(e_arg):
    return App('log', [e_arg])

def e_inc(e_arg):
    return App('+',[Cnst(1.0),e_arg])

def e_softplus(e_arg):
    return e_log(e_inc(e_exp(e_arg)))

def e_mul(e_list):
    if (len(e_list) == 0):
        return Cnst(0.0)
    elif (len(e_list) == 1):
        return e_list[0]
    elif (len(e_list) == 2):
        return App('*', e_list)
    else:
        res_tail = e_mul(e_list[1:])
        return App('*', [e_list[0], res_tail])

def e_add(e_list):
    if (len(e_list) == 0):
        return Cnst(0.0)
    elif (len(e_list) == 1):
        return e_list[0]
    elif (len(e_list) == 2):
        return App('+', e_list)
    else:
        res_tail = e_add(e_list[1:])
        return App('+', [e_list[0], res_tail])

# Global prameters
alpha1 = Var('alpha1')
alpha2 = Var('alpha2')
beta0  = Var('beta0')
beta1  = Var('beta1')
sigma1 = Var('sigma1')
sigma2 = Var('sigma2')
sigmav = Var('sigmav')

# Intialization
def e0(e_cont):
    # Local parameters
    a0 = Var('a0')
    b0 = Var('b0')
    c0 = Var('c0')
    d0 = Var('d0')
    f0 = Var('f0')

    # Expression
    res =\
        Let(alpha1, Cnst( 1.406),
        Let(alpha2, Cnst(-0.622),
        Let(beta0,  Cnst( 0.210),
        Let(beta1,  Cnst(-0.312),
        Let(sigma1, Cnst( 0.023),
        Let(sigma2, Cnst( 0.112),
        Let(sigmav, Cnst( 0.002),
        Let(a0,     Cnst(0.0),
        Let(b0,     Cnst(0.0),
        Let(c0,     Cnst(0.0),
        Let(d0,     Cnst(0.0),
        Let(f0,     Sample(Cnst(0.0),Cnst(1.0)),
        e_cont))))))))))))
    return res

def ei(i, e_cont):
    a_prev = Var('a%d' % (i-1))
    b_prev = Var('b%d' % (i-1))
    c_prev = Var('c%d' % (i-1))
    d_prev = Var('d%d' % (i-1))
    f_prev = Var('f%d' % (i-1))

    a_curr = Var('a%d' % i)
    b_curr = Var('b%d' % i)
    c_curr = Var('c%d' % i)
    d_curr = Var('d%d' % i)
    f_curr = Var('f%d' % i)

    v_curr = Var('v%d' % i)
    w_curr = Var('w%d' % i)
    o_curr = Var('o%d' % i)


    res =\
        Let(v_curr, Sample(Cnst(0.0), sigma1),
        Let(w_curr, Sample(Cnst(0.0), sigma2),
        Let(a_curr, e_add([e_mul([alpha1, a_prev]), e_mul([alpha2, b_prev]), v_curr]),
        Let(b_curr, a_prev,
        Let(c_curr, e_add([e_mul([beta1, c_prev]), beta0, w_curr]),
        Let(d_curr, d_prev,
        Let(f_curr, Sample(If(Linear(0.0, [(1.0,f_prev.v)]), 
                              Cnst(0.67),
                              Cnst(-0.67)),
                           Cnst(1.0)), 
        Let(o_curr, If(Linear(0.0, [(1.0,f_curr.v)]),
                       Observe('norm', [e_add([a_curr,d_curr]),sigmav], Cnst(data[i-1])),     
                       Observe('norm', [e_add([a_curr,c_curr,d_curr]),sigmav], Cnst(data[i-1]))),     
        e_cont))))))))
    return res

en = alpha1
e = unroll_loop(e0, ei, en, range(1, len(data)+1, 1))

thts_init = anp.zeros(get_num_thts(e))

optz_cfg = {'iter_n'  : 10000,
            'lr'      : 0.001,
            'sample_n_grad': 1,
            'sample_n_var' : 0}
plot_cfg = {'sample_n': 100,
            'step'    : 100}
compare = {'ours2' : 'elbo_grad_ours2.py',
           'score' : 'elbo_grad_score.py',
           'repar' : 'elbo_grad_repar.py'}
