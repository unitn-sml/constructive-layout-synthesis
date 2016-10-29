# -*- encoding: utf-8 -*-

import pymzn
import numpy as np

from cls.utils import freeze, input_x, input_star_x
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

    def __init__(self, canvas_size=12, num_tables=8, **kwargs):
        num_features = 10
        super().__init__(num_features)

        self._data = {'SIDE': canvas_size, 'N_TABLES': num_tables}
        self._phis = {}
        self._debug = kwargs['debug']

    def phi(self, x):
        _frx = freeze(x)
        if _frx in self._phis:
            return self._phis[_frx]

        _phi = pymzn.minizinc(self.phi_model,
                              data={**self._data, **input_x(x)},
                              output_vars=['phi'], serialize=True,
                              mzn_globals_dir='opturion-cpx', keep=True,
                              fzn_fn=pymzn.opturion)[0]['phi']
        self._phis[_frx] = np.array(_phi)
        return self._phis[_frx]

    def infer(self, w):
        return pymzn.minizinc(self.infer_model, data={**self._data, 'w': w},
                              output_vars=['x', 'y', 'dx', 'dy'],
                              mzn_globals_dir='opturion-cpx',
                              serialize=True, keep=True, 
                              fzn_fn=pymzn.opturion)[0]

    def improve(self, x, x_star, w, alpha=0.1):
        try:
            return pymzn.minizinc(self.improve_model,
                                  data={**self._data, **input_x(x), 'w': w,
                                        **input_star_x(x_star),
                                        'ALPHA': alpha},
                                  output_vars=['x', 'y', 'dx', 'dy'],
                                  mzn_globals_dir='opturion-cpx',
                                  serialize=True, keep=True,
                                  fzn_fn=pymzn.opturion)[0]
        except pymzn.MiniZincUnsatisfiableError:
            # when no improvement possible for noisy users
            return x

