# -*- encoding: utf-8 -*-

import pymzn
import numpy as np

from cls.utils import freeze, input_x
from cls.coactive import Problem


class Furniture(Problem):
    """Problem for coactive furniture arrangement.

    Parameters
    ----------
    canvas_size : positive int
        The size of the canvas.
    num_tables : positive int
        The number of tables.
    """

    infer_model = 'cls/furniture/infer.mzn'
    improve_model = 'cls/furniture/improve.mzn'
    phi_model = 'cls/furniture/phi.mzn'

    def __init__(self, canvas_size=100, num_tables=10, **kwargs):
        num_features = 4
        super().__init__(num_features)

        self._data = {'SIDE': canvas_size, 'N_TABLES': num_tables}
        self._phis = {}
        self._debug = kwargs['debug']

    def phi(self, x):
        _frx = freeze(x)
        if _frx in self._phis:
            return self._phis[_frx]

        phi = pymzn.minizinc(self.phi_model, data={**self._data, **input_x(x)},
                             output_vars=['phi'], serialize=True,
                             keep=self._debug, time=30000)[0]['phi']
        self._phis[_frx] = np.array(phi)
        return self._phis[_frx]

    def infer(self, w):
        return pymzn.minizinc(self.infer_model, data={**self._data, 'w': w},
                              output_vars=['x', 'y', 'dx', 'dy'],
                              serialize=True, keep=self._debug, time=30000)[0]

    def improve(self, x, w):
        return pymzn.minizinc(self.improve_model,
                              data={**self._data, **input_x(x), 'w': w},
                              output_vars=['x', 'y', 'dx', 'dy'],
                              serialize=True, keep=self._debug, time=30000)[0]
