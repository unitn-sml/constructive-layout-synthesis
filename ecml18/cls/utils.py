import re
import inspect
import logging
import argparse
import numpy as np

from itertools import product, chain
from textwrap import dedent
from numbers import Integral
from sklearn.utils import check_random_state


__all__ = ['get_class', 'get_defaults', 'array2string', 'dict2str', 'subdict',
           'freeze', 'get_logger', 'ContextFilter', 'mzn_range', 'mzn_dot',
           'dot_type', 'add_prefix', 'strip_prefix', 'parse_remainder']


""" Wrangling utilities """

def get_class(name, defmod):
    """Finds a class.

    Search a class from its fully qualified name, e.g. 'weaver.domains.Class'.
    If the class name is not fully qualified, e.g. 'Class', it is searched
    inside the default module.

    Parameter
    ---------
    name : str
        The fully qualified name of the class or the name of the class or the
        class name in the default module.
    defmod : str
        The default module where to search the class if not fully qualified.
    """
    if '.' not in name:
        module = __import__(defmod, fromlist=[name])
        return getattr(module, name)
    ns = name.split('.')
    name = ns[-1]
    module = __import__(ns[:-1], fromlist=[name])
    return getattr(module, name)


def get_defaults(func):
    """Gets the default values of the keyword arguments a function.

    Parameter
    ---------
    func : function
        The function to get the default values of.
    """
    spec = inspect.getfullargspec(func)
    defaults = {}
    if spec.defaults:
        defs = spec.defaults
        args = spec.args
        for i in range(len(defs)):
            defaults[args[i + len(args) - len(defs)]] = defs[i]
    if spec.kwonlydefaults:
        defaults = {**defaults, **spec.kwonlydefaults}
    return defaults


def array2string(a):
    """Pretty formatter of a numpy array."""
    return np.array2string(a, max_line_width=np.inf, separator=',',
        precision=None, suppress_small=None).replace('\n', '')


def dict2str(d):
    """Pretty formatter of a dictionary."""
    return str(sorted(d.items()))


def subdict(d, keys=None, nokeys=None):
    """Returns a subdictionary.

    Parameters
    ----------
    d : dict
        A dictionary.
    keys : list or set
        The set of keys to include in the subdictionary. If None use all keys.
    nokeys : list or set
        The set of keys to not include in the subdictionary. If None use no keys.
    """
    keys = set(keys if keys else d.keys())
    nokeys = set(nokeys if nokeys else [])
    return {k: v for k, v in d.items() if k in (keys - nokeys)}


def freeze(x):
    """Freezes a dictionary, i.e. makes it immutable and thus hashable."""
    frozen = {}
    for k, v in sorted(x.items()):
        if isinstance(v, list):
            frozen[k] = tuple(v)
        else:
            frozen[k] = v
    return frozenset(frozen.items())



""" Logging utilities """


class BraceMessage(object):
    """A logging message using braces formatting.

    Parameters
    ----------
    fmt : str
        The format string.
    args : list
        The argument list. If the list contains only one dictionary, that is
        used to get keyword arguments instead.
    kwargs : dict
        The keyword arguments dictionary.
    """
    def __init__(self, fmt, args, kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def arg_eval(arg):
        if isinstance(arg, np.ndarray):
            return array2string(arg)
        if isinstance(arg, dict):
            return dict2str(arg)
        return arg

    def __str__(self):
        kwargs = {key: self.arg_eval(arg) for key, arg in self.kwargs.items()}
        if len(self.args) == 1 and isinstance(self.args[0], dict):
            try:
                args = {k: self.arg_eval(v) for k, v in self.args[0].items()}
                return self.fmt.format(**{**args, **kwargs})
            except IndexError:
                pass
        args = [self.arg_eval(arg) for arg in self.args]
        return self.fmt.format(*args, **kwargs)


class ContextFilter(logging.Filter):
    """A filter to add contextual information in the log message.

    Parameter
    ---------
    context : str
        The context to add to the logger.
    sep : str
        The separation string between the context and the message.
    """
    def __init__(self, context, sep=', '):
        self.context = context
        self.sep = sep

    def filter(self, record):
        record.msg = self.sep.join([self.context, str(record.msg)])
        return True


class MultilineLoggerAdapter(logging.LoggerAdapter):
    """An adapter to allow multiline logging messages.
    """
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})

    def push_context(self, context, sep=', '):
        self.logger.addFilter(ContextFilter(context, sep))

    def pop_context(self):
        self.logger.filters.pop()

    def log(self, level, msg, *args,
            exc_info=None, extra=None, stack_info=False, **kwargs):
        msg = dedent(str(msg)).strip()
        for msg in msg.splitlines():
            self.logger._log(level, BraceMessage(msg, args, kwargs), (),
                             exc_info=exc_info, extra=extra,
                             stack_info=stack_info)


def get_logger(name):
    return MultilineLoggerAdapter(logging.getLogger(name))



""" Mzn utilities """


def mzn_range(lst):
    """Return the minizinc range of lst"""
    return set(range(1, len(lst) + 1))


def mzn_dot(a, b):
    """Return the minizinc definition of the dot product between a and b"""
    return 'sum(i in index_set({a}))({a}[i] * {b}[i])'.format(**locals())


def dot_type(w, feat_type):
    """Return the type of the dot product given the types of w and features"""
    if issubclass(w.dtype.type, Integral) and feat_type in {'int', 'bool'}:
        return 'int'
    return 'float'


def add_prefix(attrs, lst, prefix):
    rlst = []
    for s in lst:
        for attr in attrs:
            s = re.sub(r'\b{}\b'.format(attr), prefix + attr, s)
        rlst.append(s)
    return rlst


def strip_prefix(y, prefix):
    """Return y without the prefix."""
    return {k[len(prefix):]: v for k, v in y.items()}


def random_dnf(attributes, max_literals=4, max_clauses=4, seed=None):
    rng = check_random_state(seed)
    attr_list = list(attributes.keys())
    num_clauses = rng.randint(1, max_clauses + 1)
    formula = []
    for i in range(num_clauses):
        num_literals = rng.randint(1, max_literals + 1)
        attrs = list(range(len(attr_list)))
        clause = []
        for j in range(num_literals):
            attr_idx = rng.choice(attrs)
            attrs.remove(attr_idx)
            attr = attr_list[attr_idx]
            attr_range = attributes[attr_list[attr_idx]]
            if attr_range == 'bool':
                literal = '{}{}'.format('not ' if rng.randint(1) else '', attr)
            else:
                value = rng.randint(attr_range)
                literal = '{} {}= {}'.format(
                    attr, '=' if rng.randint(1) else '!', value + 1
                )
            clause.append(literal)
        clause = ' /\ '.join(clause)
        formula.append(clause)
    return ' \/ '.join(formula)
"""
def literal_space(attributes):
    for attr in attributes:
        if attributes[attr] == 'bool':
            yield attr
            yield 'not {}'.format(attr)
        else:
            for val in xrange(attributes[attr]):
                yield '{} == {}'.format(attr, val + 1)
                yield '{} != {}'.format(attr, val + 1)

def clause_space(attributes, max_literals):
    for n_lits in xrange(1, max_literals + 1):
        product(literals_space(attributes) for _ in range(n_lits))
        yield map(lambda c : ' /\ '.join(c), 

"""
                  
""" Argument parsing utilities """


def parse_remainder(args):
    if len(args) >= 1 and args[0] in {'--', '-'}:
        del args[0]
    if len(args) == 0:
        return {}
    if not args[0].startswith(('-', '--')):
        raise ValueError('Only optional arguments allowed')

    kwargs = {}
    curr_arg = None
    curr_val = None
    for arg in args:
        if arg.startswith(('-', '--')):
            if curr_arg:
                if not curr_val:
                    kwargs[curr_arg] = True
                else:
                    kwargs[curr_arg] = curr_val
            m = re.match('(?:--?)(?P<arg>[\w-]+)', arg)
            curr_arg = m.group('arg').replace('-', '_')
            curr_val = None
        else:
            if not curr_val:
                curr_val = arg
            elif isinstance(curr_val, list):
                curr_val.append(arg)
            else:
                curr_val = [curr_val]
                curr_val.append(arg)

    if curr_arg:
        if not curr_val:
            kwargs[curr_arg] = True
        else:
            kwargs[curr_arg] = curr_val
    return kwargs

