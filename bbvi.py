from __future__ import print_function

import importlib, pickle
import matplotlib     # trick for resolving
matplotlib.use('agg') # 'no module named _tkinter' error
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from expr import *
import util, optimizer, elbo_val


############
# settings #
############

BM_DIR = 'bm/'
RES_DIR = 'res/'
RES_EXT = '.file'
PLOT_EXT = '.png'
PRETTY_PLOT_EXT = '.pdf'


#######
# aux #
#######

def cfg2str(kind, cfg):
    if   kind == 'optz': res = 'iter=%d_lr=%g_sample=%s_samplevar=%d' %\
         (cfg['iter_n'], cfg['lr'], cfg['sample_n_grad'], cfg['sample_n_var'])
    elif kind == 'plot': res = 'sample=%d_step=%d' % (cfg['sample_n'], cfg['step'])
    else: assert(False)
    return res

def get_optz_detail(bm_fname, optz_cfg):
    return '%s%s_%s' % (RES_DIR, bm_fname, cfg2str('optz', optz_cfg))

def get_plot_detail(bm_fname, optz_cfg, plot_cfg):
    optz_detail = get_optz_detail(bm_fname, optz_cfg)
    return '%s_%s' % (optz_detail, cfg2str('plot', plot_cfg))

def save_res(optz_detail, alg_str, obj):
    res_fname = '%s_%s%s' % (optz_detail, alg_str, RES_EXT)
    with open(res_fname, 'wb') as fout:
        pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)

def load_res(optz_detail, alg_str):
    res_fname = '%s_%s%s' % (optz_detail, alg_str, RES_EXT)
    with open(res_fname, 'rb') as fin:
        return pickle.load(fin)
    assert(False)


#===== generate RES_EXT files =====#

##########
# run_bm #
##########

def run_bm(bm_fname, optz_cfg=None, verbose=True, misc=None):
    # load bm_fname
    bm = importlib.import_module(bm_fname.rsplit('.', 1)[0])
    e         = bm.e; decorate_stind(e)
    thts_init = bm.thts_init
    compare   = bm.compare
    if optz_cfg is None: optz_cfg = bm.optz_cfg

    # optz_detail
    optz_detail = get_optz_detail(bm_fname, optz_cfg)
    print('\n===== OPTZ: %s =====' % optz_detail)

    # run experiments
    for alg_str in compare:
        print('[%s] ' % alg_str, end='')
        alg = importlib.import_module(compare[alg_str].rsplit('.',1)[0])
        alg.init(e)
        misc_arg = {'misc':misc} if alg_str == 'ours2' else {}

        # run adam
        grad_func = lambda thts, e=e: alg.elbo_grad(e, thts, **misc_arg)
        thts_res = optimizer.adam(grad_func, thts_init,
                                  iter_n   = optz_cfg['iter_n'],
                                  lr       = optz_cfg['lr'],
                                  sample_n_grad = optz_cfg['sample_n_grad'],
                                  sample_n_var  = optz_cfg['sample_n_var'],
                                  verbose  = verbose,
                                  **misc_arg)

        # save res to file
        save_res(optz_detail, alg_str, thts_res)


############
# plot_res #
############

def plot_res(bm_fname, optz_cfg=None, plot_cfg=None):
    # load bm_fname
    bm = importlib.import_module(bm_fname.rsplit('.', 1)[0])
    e       = bm.e; decorate_stind(e)
    compare = bm.compare
    if optz_cfg is None: optz_cfg = bm.optz_cfg
    if plot_cfg is None: plot_cfg = bm.plot_cfg

    # {optz,plot}_detail
    optz_detail = get_optz_detail(bm_fname, optz_cfg)
    plot_detail = get_plot_detail(bm_fname, optz_cfg, plot_cfg)
    print('\n===== PLOT: %s =====' % plot_detail)

    # load res's from files
    thts_res_l = []
    alg_str_l  = []
    for alg_str in compare:
        thts_res = load_res(optz_detail, alg_str)
        thts_res = [(t,thts) for (t,thts,_,_,_) in thts_res]
        thts_res_l += [thts_res]
        alg_str_l  += [alg_str]
    
    # plot & save graph
    plot_fname = '%s%s' % (plot_detail, PLOT_EXT)
    text_fname = plot_fname[:-len(PLOT_EXT)] + RES_EXT
    objc_func = lambda thts, e=e: elbo_val.elbo_val(e, thts,
                                                    sample_n = plot_cfg['sample_n'])
    util.plot_graph(thts_res_l, objc_func,
                    plot_fname = plot_fname,
                    text_fname = text_fname,
                    legend_l = alg_str_l,
                    step = plot_cfg['step'])

    
#===== use RES_EXT file already computed =====#

###################
# print_last_thts #
###################

def print_last_thts(bm_fname, optz_cfg=None):
    # load bm_fname
    bm = importlib.import_module(bm_fname.rsplit('.', 1)[0])
    e       = bm.e; decorate_stind(e)
    compare = bm.compare
    if optz_cfg is None: optz_cfg = bm.optz_cfg

    # optz_detail
    optz_detail = get_optz_detail(bm_fname, optz_cfg)
    print('\n===== inferred thts: %s =====' % optz_detail)

    # load res's from files
    thts_l    = []
    alg_str_l = []
    for alg_str in compare:
        thts = load_res(optz_detail, alg_str)[-1][1]
        thts_l    += [thts]
        alg_str_l += [alg_str]

    # print
    print('\t%s' % ('\t\t'.join(alg_str_l)))
    for i in range(len(thts_l[0])/2):
        thts_i_float = util.flatten_list([[thts[2*i], util.softplus(thts[2*i+1])] for thts in thts_l])
        thts_i_str = ['%.3f' % v for v in thts_i_float]
        print('tht_%d(mean)\t%s' % (i+1, '\t'.join(thts_i_str[0::2])))
        print('tht_%d(std )\t%s' % (i+1, '\t'.join(thts_i_str[1::2])))
        

###############
# pretty_plot #
###############

def pretty_plot(bm_fname, optz_cfg, plot_cfg, legend_loc=4, sci_format=False, y_min=0, y_max=0):
    LEGEND = { # alg : (order in legend, label to appear)
        'ours2' : (3, r'\textsc{Ours}'),
        'repar' : (2, r'\textsc{Repar}'),
        'score' : (1, r'\textsc{Score}'),
    }
    LINE_COLOR = {
        'ours2' : 'xkcd:red', #'r', C2
        'repar' : 'xkcd:bright blue', #'b', C3
        'score' : 'xkcd:green', #'g', C5
    }
    LINE_STYLE = {
        # linestyle = : | -. | -- | -
        1 : {'linestyle' : '-'},
        2 : {'linestyle' : '--', 'dash_capstyle' : 'round'}, #, 'dashes' : [8,3]},
        3 : {'linestyle' : ':' , 'dash_capstyle' : 'round'}  #, 'dashes' : [3,3,18,3]}
    }
    FONTSIZE = 22

    # compute: data_dict
    data_dict = {}
    sample_n_grad_l = optz_cfg['sample_n_grad']

    for sample_n_grad in sample_n_grad_l:
        # load: elbos
        optz_cfg['sample_n_grad'] = sample_n_grad
        plot_detail = get_plot_detail(bm_fname, optz_cfg, plot_cfg)
        text_fname = '%s%s' % (plot_detail, RES_EXT)
        with open(text_fname, 'rb') as fin:
            res = pickle.load(fin)

        # update: data_dict
        for (alg, xsys) in zip(res['legend_l'], res['xsys_l']):
            if alg not in data_dict: data_dict[alg] = []
            data_dict[alg] += [('%s' % LEGEND[alg][1], xsys)]
            # data_dict[alg] += [('$\\mathrm{%s}\ (N=%d)$' % (LEGEND[alg][1], sample_n_grad), xsys)]
            # '$\\mathrm{%s}\ (L=%d)$'  or  '%s (L=%d)'

    ############# plot #############
    optz_cfg['sample_n_grad'] = str(sample_n_grad_l)[1:-1].replace(' ','')
    plot_detail = get_plot_detail(bm_fname, optz_cfg, plot_cfg)
    plot_fname = '%s_pretty%s' % (plot_detail, PRETTY_PLOT_EXT)
    plt.switch_backend('agg')
    plt.style.use('classic')
    plt.rc('text', usetex = True)
    plt.rc('font', size=FONTSIZE, family='serif', serif='Computer Modern')
    # for bug fixing: matplotlib inserts \mathdefault{...} which is an useless, undefined command.
    plt.rc('text.latex', preamble = r'\newcommand{\mathdefault}[1]{{#1}}')
    fig, ax = plt.subplots(figsize=(8,6))  # (8,5) <--- size
    
    # legend
    for alg in sorted(data_dict, key=lambda _alg: LEGEND[_alg][0]):
        cnt = len(data_dict[alg])
        for (legend, (xs, ys)) in data_dict[alg]:
            if cnt > 1: legend = ''
            ax.plot(xs, ys, label=legend, linewidth=2, color=LINE_COLOR[alg], **LINE_STYLE[cnt])
            cnt -= 1
    ax.legend(loc=legend_loc, framealpha=0.8, fontsize=FONTSIZE, edgecolor='w')

    # axis
    ax.set_xlim(0, optz_cfg['iter_n'])
    if y_min < y_max: ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r'Iteration')
    ax.set_ylabel(r'ELBO')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    """
    # formatting axis numbers
    msf = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    format_sci = lambda x,pos : "${}$".format(msf._formatSciNotation('%1.10e' % x))
    format_int = lambda x,pos : "${}$".format(int(x))
    format_x = format_int
    format_y = format_sci if sci_format == True else format_int
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_x))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_y))
    """

    # save
    plt.tight_layout()
    plt.savefig(plot_fname)
    print('DONE: %s' % plot_fname)


############
# plot_var #
############

def plot_var(bm_fname, optz_cfg, plot_cfg):
    # load bm_fname
    bm = importlib.import_module(bm_fname.rsplit('.', 1)[0])
    compare = bm.compare

    # optz_detail
    optz_detail = get_optz_detail(bm_fname, optz_cfg)
    print('\n===== plot_var: %s =====' % optz_detail)

    # init plot
    plt.switch_backend('agg')
    fig, ax = plt.subplots(2, figsize=(8,6))
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    # load res's from files
    for alg_str in compare:
        if alg_str == 'score': continue
        thts_res = load_res(optz_detail, alg_str)
        xs1, var1 = zip(*[(t,var1) for (t,_,var1,_,_) in thts_res])
        xs2, var2 = zip(*[(t,var2) for (t,_,_,var2,_) in thts_res])
        ax[0].plot(xs1, var1, label=alg_str+' (var1)')
        ax[1].plot(xs2, var2, label=alg_str+' (var2)')

    # save
    plot_fname = '%s_var%s' % (optz_detail, PLOT_EXT)
    ax[0].legend(loc=1)
    ax[1].legend(loc=1)
    plt.savefig(plot_fname)
    print('DONE: %s' % plot_fname)


#===== others =====#

####################
# compute_elbo_val #
####################

def compute_elbo_val(bm_fname, thts_l, sample_n):
    # load bm_fname
    bm = importlib.import_module(bm_fname.rsplit('.', 1)[0])
    e  = bm.e; decorate_stind(e)

    # {bm,plot}_detail
    bm_detail  = '%s' % bm_fname
    val_detail = 'sample=%d' % sample_n
    print('\n===== ELBO_VAL: %s // %s =====' % (bm_detail, val_detail))

    # compute elbo_val
    for (name, thts) in thts_l:
        res = elbo_val.elbo_val(e, thts, sample_n=sample_n)
        print('[%s] elbo = %g' % (name, res))
        print_thts(e, thts)
