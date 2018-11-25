from __future__ import print_function

import abc, six
import autograd.numpy as anp
import autograd.scipy as ascipy

import util


########
# Expr #
########

@six.add_metaclass(abc.ABCMeta)
class Expr(object):
    @abc.abstractmethod
    def __str__(self): raise NotImplementedError()


############################
# Cnst, Var, Linear        #
# App, If, Let             #
# Sample, Fsample, Observe #
############################

class Cnst(Expr):
    # c
    def __init__(self, c): assert(isinstance(c, float)); self.c = c
    def __str__(self): return ("%g" % self.c)
    
class Var(Expr):
    # v
    def __init__(self, v): assert(isinstance(v, str)); self.v = v
    def __str__(self): return ("%s" % self.v)

class Linear(Expr):
    # [[ Linear(c0, [(c1,v1), ..., (cn,vn)]) ]]
    # = c0 + c1*(value of Var(v1)) + ... + cn*(value of Var(vn)),
    # where ci : float, and vi : str.
    # Here vi is the name of a variable storing a [f]sampled result,
    # i.e., (let Var(vi) ([f]sample ...) (...)) is contained in the whole expression.
    def __init__(self, c0, cv_l):
        assert(isinstance(c0, float));   self.c0   = c0
        assert(_is_floatstr_list(cv_l)); self.cv_l = cv_l
        assert(len(cv_l) > 0)
        
    def __str__(self):
        res = "%g" % self.c0
        for i in range(len(self.cv_l)):
            res += " + %g*%s" % (self.cv_l[i][0], self.cv_l[i][1])
        return ("(%s)" % res)

class App(Expr): 
    # (op e1 ... en)
    def __init__(self, op, args):
        assert(op in App.OP_DICT[len(args)]); self.op   = op
        assert(_is_expr_list(args));          self.args = args

    def __str__(self):
        return ("(%s %s)" 
                % (self.op, 
                   _expr_list_to_str(self.args)))

    OP_DICT = [{},
        # unary op
        {'-'   : anp.negative,
         'exp' : anp.exp,
         'log' : anp.log,
         'sqrt': anp.sqrt,
         'sin' : anp.sin,
         'cos' : anp.cos,
         'tan' : anp.tan},
        # binary op
        {'+'   : anp.add,
         '-'   : anp.subtract,
         '*'   : anp.multiply,
         '/'   : anp.divide,
         'pow' : anp.power}]

class If(Expr):
    # (if (e1 > 0) e2 e3)
    def __init__(self, e1, e2, e3):
        assert(isinstance(e1, Expr)); self.e1 = e1
        assert(isinstance(e2, Expr)); self.e2 = e2
        assert(isinstance(e3, Expr)); self.e3 = e3

    def __str__(self):
        return ("(if (%s > 0) %s %s)"
                % (self.e1.__str__(),
                   self.e2.__str__(),
                   self.e3.__str__()))
    
class Let(Expr): 
    # (let [v1 e1] e2)
    def __init__(self, v1, e1, e2):
        assert(isinstance(v1, Var )); self.v1 = v1
        assert(isinstance(e1, Expr)); self.e1 = e1
        assert(isinstance(e2, Expr)); self.e2 = e2

    def __str__(self):
        return ("(let [%s %s] %s)"
                % (self.v1.__str__(),
                   self.e1.__str__(),
                   self.e2.__str__()))

class Sample(Expr):
    # (sample e1 e2)
    # for now, assume normal distribution.
    # e1 is mean, e2 is standard deviation.
    def __init__(self, e1, e2):
        assert(isinstance(e1, Expr)); self.e1 = e1
        assert(isinstance(e2, Expr)); self.e2 = e2

    def __str__(self):
        return ("(sample %s %s)"
                % (self.e1.__str__(),
                   self.e2.__str__()))

class Fsample(Expr):
    # (fsample e1 e2)
    # for now, assume normal distribution.
    # e1 is mean, e2 is standard deviation.
    def __init__(self, e1, e2):
        assert(isinstance(e1, Expr)); self.e1 = e1
        assert(isinstance(e2, Expr)); self.e2 = e2

    def __str__(self):
        return ("(fsample %s %s)"
                % (self.e1.__str__(),
                   self.e2.__str__()))
    
class Observe(Expr):
    # (observe dstr args c1), or
    # (observe e1 e2 c1)  --- for backward compatibility
    # # for now, assume normal distribution.
    # # e1 is mean, e2 is standard deviation, c1 is observed value.
    def __init__(self, dstr, args, c1):
        if isinstance(dstr, str):
            assert(dstr in Observe.DSTR_DICT); self.dstr = dstr
            assert(_is_expr_list(args));       self.args = args
            assert(isinstance(c1, Cnst));      self.c1   = c1
            assert(len(args) > 0)
        else: # --- for backward compatibility
            e1 = dstr; e2 = args
            assert(isinstance(e1, Expr))
            assert(isinstance(e2, Expr)) 
            assert(isinstance(c1, Cnst))
            self.dstr = 'norm'
            self.args = [e1, e2]
            self.c1   = c1
    """
    def __init__(self, e1, e2, c1):
        assert(isinstance(e1, Expr)); self.e1 = e1
        assert(isinstance(e2, Expr)); self.e2 = e2
        assert(isinstance(c1, Cnst)); self.c1 = c1
    """

    def __str__(self):
        return ("(observe (%s %s) %s)"
                % (self.dstr,
                   _expr_list_to_str(self.args),
                   self.c1.__str__()))

    DSTR_DICT = {
        """
        0) Distributions in scipy.stats and autograd.scipy.stats
             https://docs.scipy.org/doc/scipy/reference/stats.html
             https://github.com/scipy/scipy/blob/master/scipy/stats/_continuous_distns.py
             https://github.com/scipy/scipy/blob/master/scipy/stats/_discrete_distns.py
             https://github.com/HIPS/autograd/tree/master/autograd/scipy/stats
        1) The semantics of loc & scale is as follows:
             pdf(x, args, loc, scale) = pdf((x-loc)/scale, args, 0, 1) / scale
        2) pdf
             beta: pdf(x,a,b) = Gamma(a+b) x^{a-1} (1-x)^{b-1} / (Gamma(a) Gamma(b))
             chi2: pdf(x,df) = (x/2)^{df/2-1} exp{-x/2} / (2 Gamma(df/2))
             gamma: pdf(x,a) = x^{a-1} exp{-x} / Gamma(a)
             norm: pdf(x,loc,scale) = exp{-(x-loc)^2/(2scale^2)} / (sqrt{2pi scale^2})
             t: pdf(x,df) = Gamma((df+1)/2) / ( (sqrt{pi df} (df/2) (1+x^2/df)^{(df+1)/2} )
             expon: pdf(x, lam) = lam * e^{-lam * x}
             lognorm: pdf(x, sigma, mu) = exp{-(ln{x}-mu)^2/(2sigma^2)} / (sqrt{2pi} sigma x)
           pmf
             poisson: pmf(k, mu) = exp(-mu) mu^k / k!
             bernoulli: pmf(k, p) = 1-p if k=0, p if k=1
             binom: pmf(k, n, p) = choose(n,k) * p**k * (1-p)**(n-k)
             geom: pmf(k, p) = p*(1-p)^{k-1} for k>=1
        """
        # continuous
        'beta'      : ascipy.stats.beta .logpdf,  # x, a, b
        'chi2'      : ascipy.stats.chi2 .logpdf,  # x, df
        'gamma'     : ascipy.stats.gamma.logpdf,  # x, a
        'norm'      : ascipy.stats.norm .logpdf,  # x, loc, scale
        't'         : ascipy.stats.t    .logpdf,  # x, df, loc, scale
        'expon'     : util.expon_logpdf,          # x, lam
        'lognorm'   : util.lognorm_logpdf,        # x, sigma, mu
        # discrete
        'poisson'   : ascipy.stats.poisson.logpmf,  # k, mu
        'bernoulli' : util.bernoulli_logpdf,        # k, p
        'binom'     : util.binom_logpdf,            # k, n, p
        'geom'      : util.geom_logpdf,             # k, p
        # # multivariate
        # 'dirichlet' : ascipy.stats.dirichlet.logpdf,  # x, alpha
    }

    
#############
# Functions #
#############

def _is_expr_list(es): 
    is_expr = lambda e: isinstance(e, Expr)
    return (isinstance(es, list) and all(map(is_expr, es)))

def _is_floatstr_list(cvs):
    is_floatstr = lambda (c,v): isinstance(c, float) and isinstance(v, str)
    return (isinstance(cvs, list) and all(map(is_floatstr, cvs)))

def _expr_list_to_str(es, sep=" "):
    return sep.join(map(str, es))

def get_num_thts(e):
    return get_num_samples(e) * 2

def get_num_samples(e, cnt=0):
    if   isinstance(e, Cnst):   pass
    elif isinstance(e, Var):    pass
    elif isinstance(e, Linear): pass

    elif isinstance(e, If):
        cnt = get_num_samples(e.e1, cnt)
        cnt = get_num_samples(e.e2, cnt)
        cnt = get_num_samples(e.e3, cnt)

    elif isinstance(e, App):
        for ei in e.args:
            cnt = get_num_samples(ei, cnt)

    elif isinstance(e, Let):
        cnt = get_num_samples(e.e1, cnt)
        cnt = get_num_samples(e.e2, cnt)

    elif isinstance(e, Sample):
        cnt += 1  # add 1
        cnt = get_num_samples(e.e1, cnt)
        cnt = get_num_samples(e.e2, cnt)

    elif isinstance(e, Fsample):
        cnt = get_num_samples(e.e1, cnt)
        cnt = get_num_samples(e.e2, cnt)

    elif isinstance(e, Observe):
        for ei in e.args:
            cnt = get_num_samples(ei, cnt)
        cnt = get_num_samples(e.c1, cnt)

    else: assert(False)
    return cnt

def  decorate_stind(e): return _decorate_stind(e, {'epss':0, 'thts':0})
def _decorate_stind(e, cnt):
    if   isinstance(e, Cnst):   pass
    elif isinstance(e, Var):    pass
    elif isinstance(e, Linear): pass

    elif isinstance(e, App):
        for ei in e.args:
            cnt = _decorate_stind(ei, cnt)

    elif isinstance(e, If):
        cnt = _decorate_stind(e.e1, cnt)
        cnt = _decorate_stind(e.e2, cnt)
        cnt = _decorate_stind(e.e3, cnt)

    elif isinstance(e, Let):
        cnt = _decorate_stind(e.v1, cnt)
        cnt = _decorate_stind(e.e1, cnt)
        cnt = _decorate_stind(e.e2, cnt)

    elif isinstance(e, Sample):
        # ASSUME: gaussian distribution w/ 2 params
        e.stind = {'epss' : cnt['epss'],
                   'thts' : cnt['thts']}  # record thtind
        cnt['epss'] = cnt['epss'] + 1
        cnt['thts'] = cnt['thts'] + 2
        cnt = _decorate_stind(e.e1, cnt)
        cnt = _decorate_stind(e.e2, cnt)

    elif isinstance(e, Fsample):
        cnt = _decorate_stind(e.e1, cnt)
        cnt = _decorate_stind(e.e2, cnt)

    elif isinstance(e, Observe):
        for ei in e.args:
            cnt = _decorate_stind(ei, cnt)
        cnt = _decorate_stind(e.c1, cnt)

    else: assert(False)
    return cnt

def print_thts(e, thts, cnt=1):
    if   isinstance(e, Cnst):   pass
    elif isinstance(e, Var):    pass
    elif isinstance(e, Linear): pass
    
    elif isinstance(e, If):
        (thts, cnt) = print_thts(e.e1, thts, cnt)
        (thts, cnt) = print_thts(e.e2, thts, cnt)
        (thts, cnt) = print_thts(e.e3, thts, cnt)

    elif isinstance(e, App):
        for ei in e.args:
            (thts, cnt) = print_thts(ei, thts, cnt)

    elif isinstance(e, Let):
        (thts, cnt) = print_thts(e.e1, thts, cnt)
        (thts, cnt) = print_thts(e.e2, thts, cnt)

    elif isinstance(e, Sample):
        # print tht
        print('tht_%d: (mn, sd) = (%g, %g)' % (cnt, thts[0], util.softplus(thts[1])))
        (thts, cnt) = (thts[2:], cnt+1)
        (thts, cnt) = print_thts(e.e1, thts, cnt)
        (thts, cnt) = print_thts(e.e2, thts, cnt)

    elif isinstance(e, Fsample):
        (thts, cnt) = print_thts(e.e1, thts, cnt)
        (thts, cnt) = print_thts(e.e2, thts, cnt)

    elif isinstance(e, Observe):
        for ei in e.args:
            (thts, cnt) = print_thts(ei, thts, cnt)
        (thts, cnt) = print_thts(e.c1, thts, cnt)

    else: assert(False)
    return (thts, cnt)

def unroll_loop(e0, ei, en, i_l):
    """
    Args:
      - e0 : Expr -> Expr
      - ei : int * Expr -> Expr
      - en : Expr
      - i_l : list or generator
    Returns:
      - e0( ei(i1, ei(i2, ... ei(in, en) ...)) )
    where i_l = [i1, i2, ..., in]
    """
    res = en
    for i in reversed(i_l):
        res = ei(i, res)
    res = e0(res)
    return res
