from __future__ import print_function

import logging, sys, os
import importlib
import numpy as np

from expr import *
import util, bbvi
        
        
########
# test #
########

def test():
    # run benchmarks
    """
    bbvi.run_bm  ('bm1.py')
    bbvi.plot_res('bm1.py')
    bbvi.run_bm  ('bm2.py')
    bbvi.plot_res('bm2.py')
    bbvi.run_bm  ('bm5.py')
    bbvi.plot_res('bm5.py')
    bbvi.run_bm  ('bm3-2.py')
    bbvi.plot_res('bm3-2.py')
    bbvi.run_bm  ('bm4-3.py', verbose=True)
    bbvi.plot_res('bm4-3.py')
    bbvi.run_bm  ('bm7-2.py')
    bbvi.plot_res('bm7-2.py')
    bbvi.run_bm  ('bm8.py')
    bbvi.plot_res('bm8.py')
    bbvi.run_bm  ('bm8-2.py')
    bbvi.plot_res('bm8-2.py')

    bbvi.run_bm  ('bm4-3-a.py')
    bbvi.plot_res('bm4-3-a.py')
    bbvi.run_bm  ('bm4-3-b.py')
    bbvi.plot_res('bm4-3-b.py')
    bbvi.run_bm  ('bm4-3-c.py')
    bbvi.plot_res('bm4-3-c.py')
    bbvi.run_bm  ('bm4-3-d.py')
    bbvi.plot_res('bm4-3-d.py')
    bbvi.run_bm  ('bm4-4.py')
    bbvi.plot_res('bm4-4.py')
    bbvi.run_bm  ('bm4-4-b.py')
    bbvi.plot_res('bm4-4-b.py')
    bbvi.run_bm  ('bm4-5.py')
    bbvi.plot_res('bm4-5.py')
    bbvi.run_bm  ('bm4-5-a.py')
    bbvi.plot_res('bm4-5-a.py')
    bbvi.run_bm  ('bm4-5-b.py')
    bbvi.plot_res('bm4-5-b.py')
    """
    
    # plot results
    """
    bbvi.plot_res('bm3-2.py', plot_cfg={'sample_n':10000, 'step':100})
    """
    
    # compute elbo values
    """
    thts_l = [
        ('ours2',
         [3.35079,   util.softplus_inv(0.163907),
          2.25913,   util.softplus_inv(0.373976),
          2.41067,   util.softplus_inv(0.251695)]),
        ('ours2-modified',
         [3.35079,   util.softplus_inv(0.163907),
          2.25913,   util.softplus_inv(0.373976),
          0.0245139,   util.softplus_inv(0.988494)]),
        ('repar',
         [3.35020,   util.softplus_inv(0.209160),
          3.30736,   util.softplus_inv(0.222924),
          0.0245139, util.softplus_inv(0.988494)])
    ]
    bbvi.compute_elbo_val('bm4-2.py', thts_l, 100)
    """
    """
    mu_l = np.arange(-0.3, 0., 0.01)
    thts_l = [('%g'%mu, [mu, util.softplus_inv(0.3)]) for mu in mu_l]
    bbvi.compute_elbo_val('bm3-2.py', thts_l, 5000)
    """
    

#############
# do_exprmt #
#############

def do_exprmt(bm_fname, optz_cfg, plot_cfg, actions, misc): # expriments for NIPS'18
    # parse actions
    action_l = actions.split(',')
    (to_run, to_plot) = (False, False)
    for action in action_l:
        if   action == 'run' : to_run  = True
        elif action == 'plot': to_plot = True
        else: assert(False)

    # run, plot
    if to_run : bbvi.run_bm  (bm_fname, optz_cfg=optz_cfg, misc=misc)
    if to_plot: bbvi.plot_res(bm_fname, optz_cfg=optz_cfg, plot_cfg=plot_cfg)


########
# main #
########
if __name__ == "__main__":

    #########
    # setup #
    #########
    # Set 'level' to
    # 1) 'logging.DEBUG' to print debug msgs, or
    # 2) 'logging.INFO' to disable debug pr
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    # increment recursion depth
    # (to prevent "maximum recursion depth exceeded" error from multiprocessing module)
    sys.setrecursionlimit(10000)
    
    # add bbvi.BM_DIR to sys.path
    sys.path.append(os.path.dirname(__file__) + bbvi.BM_DIR)

    ####################
    # argument parsing #
    ####################
    # print --help info
    if len(sys.argv) not in [10, 11]:
        print('python main_run.py [res_dir:str] [bm_fname:str] [optz_cfg] [plot_cfg] [actions] [misc]')
        print('  [optz_cfg] = [iter_n:int] [lr:float] [sample_n_grad:int] [sample_n_var:int]')
        print('  [plot_cfg] = [sample_n:int] [step:int]')
        print('  [actions:str] = run,plot | run | plot')
        print('  [misc:str] = | sep_var')
        print('  NOTE: sample_n_var == 0 implies sample_n_var = sample_n_grad\n')
        exit(1)
    
    # set & mkdir bbvi.RES_DIR
    assert(sys.argv[1][-1] == '/'); bbvi.RES_DIR = sys.argv[1]
    if not os.path.exists(bbvi.RES_DIR): os.makedirs(bbvi.RES_DIR)

    # get others
    bm_fname = sys.argv[2]
    optz_cfg = {'iter_n'   : int  (sys.argv[3]),
                'lr'       : float(sys.argv[4]),
                'sample_n_grad' : int  (sys.argv[5]),
                'sample_n_var'  : int  (sys.argv[6])}
    plot_cfg = {'sample_n' : int  (sys.argv[7]),
                'step'     : int  (sys.argv[8])}
    actions = sys.argv[9]
    if len(sys.argv) == 10: misc = None
    if len(sys.argv) == 11:
        if sys.argv[10] == 'sep_var': misc = 'sep_var'
        else: assert(False)
    

    #######
    # run #
    #######
    # test()
    do_exprmt(bm_fname, optz_cfg, plot_cfg, actions, misc)
