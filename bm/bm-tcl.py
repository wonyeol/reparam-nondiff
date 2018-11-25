# this file is based on bm7-2.py

"""
Model for Thermosttically Controlled Loads (from Soudjani et al.'s QEST'17 paper)
"""

import autograd.numpy as anp
from expr import *

theta_s = 20.0
theta_a = 32.0
delta_d = 4.0  # 0.25
P_rate = 14.0 
R = 1.5
C = 10.0
invCR = 1. / (C * R)
sigma0 = 0.2
sigma1 = 0.22
RP_rate = R * P_rate
sqrt_time_step = 2.0

theta_lower = theta_s - (delta_d / 2.0)
theta_upper = theta_s + (delta_d / 2.0)

dataset = [ # dataset in NIPS18model-tcl.clj
    [[0, 19.999393052411417], 19.793315726903277],

    [[0, 20.458453540109634], 20.57447277591924 ],
    [[0, 20.985832899298813], 23.318166935159923],
    [[0, 22.38768406603287 ], 21.732074976392273],
    [[1, 23.550508079883446], 23.419968943481056],
    [[1, 22.11729366445944 ], 23.45150129902584 ],
    
    [[1, 21.064619167851312], 20.849568706918458],
    [[1, 19.832404549285794], 20.547137185277485],
    [[1, 19.105507091196795], 18.211539341750292],
    [[1, 18.555867908555115], 18.505305155632083],
    [[1, 17.565304931813326], 16.33716312726332 ],
    
    [[0, 16.784272508644587], 16.381854726212698],
    [[0, 18.498312350185266], 19.404931063949995],
    [[0, 19.427764633078546], 19.042584619340943],
    [[0, 20.160476022468195], 20.646440471252035],
    [[0, 21.550786234122324], 22.021474886012076],
    
    [[0, 22.051865745051266], 21.767275215344945],
    [[1, 22.856132484853177], 22.59452466844986 ],
    [[1, 21.756492085507755], 20.437738111776696],
    [[1, 20.220631832476116], 21.478557643265425],
    [[1, 19.617488064098556], 18.513733108301032]
] 
obs = map(lambda x: x[1], dataset) # dataset-obs in NIPS18model-tcl.clj
qs_gt = map(lambda x: x[0][0], dataset)

def e0(e_last):
    q_cur = 'q0'
    theta_cur = 'theta0'
    obs_cur = 'obs0'
    res =\
        Let(Var(q_cur), Cnst(0.),
        Let(Var(theta_cur), Sample(Cnst(20.), Cnst(0.001)),
        Let(Var(obs_cur), Observe('norm', [Var(theta_cur), Cnst(1.)], Cnst(obs[0])),
        e_last )))
    return res

def ei(i, e_last):
    q_prev = 'q%d' % (i-1)
    theta_prev = 'theta%d' % (i-1)
    q_cur_noise = 'q%d-noise' % i
    q_cur = 'q%d' % i
    b_cur = 'b%d' % i
    sigma_cur = 'sigma%d' % i
    theta_cur = 'theta%d' % i
    obs_cur = 'obs%d' % i
    res =\
        Let(Var(q_cur_noise), Sample(If(Linear(theta_lower, [(-1., theta_prev)]), 
                                        Cnst(0.0),
                                        If(Linear(-theta_upper, [(1., theta_prev)]),
                                           Cnst(1.0),
                                           Var(q_prev) )), 
                                     Cnst(0.001)),
        Let(Var(q_cur), If(Linear(-0.5, [(1., q_cur_noise)]), Cnst(1.), Cnst(0.)),
        Let(Var(b_cur), App('*', [Cnst(invCR), 
                                  App('-', [Cnst(theta_a), 
                                            App('+', [Var(theta_prev),
                                                      App('*', [Var(q_cur), Cnst(RP_rate)]) ])])]),
        Let(Var(sigma_cur), If(Linear(-0.5, [(1., q_cur_noise)]), Cnst(sigma1), Cnst(sigma0)),
        Let(Var(theta_cur), Sample(App('+', [Var(theta_prev), Var(b_cur)]), 
                                   App('*', [Var(sigma_cur), Cnst(sqrt_time_step)])),
        Let(Var(obs_cur), Observe('norm', [Var(theta_cur), Cnst(1.)], Cnst(obs[i])),
        e_last ))))))
    return res

en = Var('theta20')
e = unroll_loop(e0, ei, en, range(1, 21, 1))

thts_init = anp.zeros(get_num_thts(e))
thts_init[0] = 20.
thts_init[1] = util.softplus_inv(0.001)
for i in range(len(obs)-1):
    thts_init[2+4*i  ] = 0.5
    thts_init[2+4*i+1] = util.softplus_inv(0.001)
    thts_init[2+4*i+2] = obs[i]
    thts_init[2+4*i+3] = util.softplus_inv(sigma0*sqrt_time_step)

optz_cfg = {'iter_n'  : 10000,
            'lr'      : 0.001,
            'sample_n_grad': 1,
            'sample_n_var' : 0}
plot_cfg = {'sample_n': 100,
            'step'    : 100}
compare = {'ours2' : 'elbo_grad_ours2.py',
           'score' : 'elbo_grad_score.py',
           'repar' : 'elbo_grad_repar.py'}
