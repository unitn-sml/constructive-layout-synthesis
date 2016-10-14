# -*- encoding: utf-8 -*-

import logging
import numpy as np


"""Logging utilities"""

class NpMessage(object):
    def __init__(self, fmt, args, np_precision=None, np_suppress_small=None):
        self.fmt = fmt
        self.args = args
        self.np_precision = np_precision
        self.np_suppress_small = np_suppress_small

    def __str__(self):
        args = [self.str_array(arg) if isinstance(arg, np.ndarray) else arg
                for arg in self.args]
        return self.fmt.format(*args)

    def str_array(self, a):
        s = np.array2string(a, max_line_width=np.inf, separator=',',
                            precision=self.np_precision,
                            suppress_small=self.np_suppress_small)
        return s.replace('\n', '')


class BracesAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra=None,
                 np_precision=None, np_suppress_small=None):
        super().__init__(logger, extra or {})
        self.np_precision = np_precision
        self.np_suppress_small = np_suppress_small

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            m = NpMessage(msg, args, np_precision=self.np_precision, 
                        np_suppress_small=self.np_suppress_small)
            self.logger._log(level, m, (), **kwargs)


