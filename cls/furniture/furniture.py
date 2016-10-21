# -*- encoding: utf-8 -*-

import pymzn
import numpy as np

from cls.utils import freeze, input_x
from cls.coactive import Problem


class Furniture(Problem):
    """Problem for coactive furniture arrangement.

    Parameters
    ----------
    num_features : positive int
        The number of features in :math:`\phi`
    """

    phi_model = 'phi.mzn'
    infer_model = 'infer.mzn'
    improve_model = 'improve.mzn'

    def __init__(self, canvas_size=100, num_tables=20, **kwargs):
        num_features = 4
        super().__init__(num_features)

        self._data = {'SIDE': canvas_size, 'N_TABLES': num_tables}
        self._phis = {}

    def phi(self, x):
        _frx = freeze(x)
        if _frx in self._phis:
            return self._phis[_frx]

        phi = pymzn.minizinc(phi_model, data={**self._data, **input_x(x)},
                             output_vars=['phi'])[0]['phi']
        self._phis[_frx] = np.array(phi)
        return self._phis[_frx]

    def infer(self, w):
        return pymzn.minizinc(infer_model, data={**self._data, 'w': w},
                              output_vars=['x', 'y', 'dx', 'dy'])[0]

    def improve(self, x, w):
        return pymzn.minizinc(improve_model,
                              data={**self._data, **input_x(x), 'w': w},
                              output_vars=['x', 'y', 'dx', 'dy'])[0]
