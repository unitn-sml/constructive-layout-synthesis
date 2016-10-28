# -*- encoding: utf-8 -*-

import logging
import numpy as np


"""Logging utilities"""

def array2str(a):
    s = np.array2string(a, max_line_width=np.inf, separator=',',
                        precision=None, suppress_small=None)
    return s.replace('\n', '')


def x2str(x):
    return sorted(x.items())


class LazyMessage(object):
    def __init__(self, fmt, args):
        self.fmt = fmt
        self.args = args

    def __str__(self):
        _args = [arg() if hasattr(arg, '__call__') else arg
                 for arg in self.args]
        return self.fmt.format(*_args)

class BracesAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger._log(level, LazyMessage(msg, args), (), **kwargs)


def get_logger(name):
    return BracesAdapter(logging.getLogger(name))


"""Wrangling utilities"""

def freeze(x):
    """Freezes a dictionary, i.e. makes it immutable and thus hashable."""
    frozen = {}
    for k, v in x.items():
        if isinstance(v, list):
            frozen[k] = tuple(v)
        else:
            frozen[k] = v
    return frozenset(frozen.items())


def input_x(x):
    return {'input_' + k: v for k, v in x.items()}

def input_star_x(x):
    return {'input_star_' + k: v for k, v in x.items()}


def subdict(d, nokeys):
    return {k: v for k, v in d.items() if k in d.keys() - set(nokeys)}

