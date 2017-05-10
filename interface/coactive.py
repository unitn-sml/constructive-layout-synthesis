
import pymzn
import numpy as np

from utils import freeze, subdict


class Domain:

    def __init__(self, mzn_phi, mzn_infer, mzn_improve, num_features):
        self.mzn_phi = mzn_phi
        self.mzn_infer = mzn_infer
        self.mzn_improve = mzn_improve
        self.num_features = num_features
        self._phis = {}
        self._infers = {}
        self._improves = {}

    @staticmethod
    def inputize(x, target_keys):
        x_input = {key: val for key, val in x.items() if key not in target_keys}
        for k in target_keys:
            x_input['input_' + k] = x[k]
        return x_input

    def phi(self, x, y):
        _frx = freeze(x), freeze(y)
        if _frx not in self._phis:
            ykeys = ['x', 'y', 'dx', 'dy']
            _phi = pymzn.minizinc(self.mzn_phi,output_vars=['phi'],
                    data={**self.inputize(subdict(y, ykeys), ykeys), **x},
                    solver=pymzn.opturion)[0]['phi']
            self._phis[_frx] = np.array(_phi, dtype=np.float64)
        return self._phis[_frx]

    def infer(self, x, w):
        _frx = freeze(x), freeze(w)
        if _frx not in self._infers:
            _argmax = pymzn.minizinc(self.mzn_infer, data={**x, 'w': w}, solver=pymzn.opturion)[0]
            self._infers[_frx] = _argmax
        return self._infers[_frx]

    def improve(self, x, phi, changed):
        """Returns an object with the given phi.

        This is used to get a new object after the user changes some feature.
        """
        _frx = freeze(x), freeze(phi)
        if _frx not in self._improves:
            _impr = pymzn.minizinc(self.mzn_improve, data={**x, 'input_phi': phi,
                                       'changed': changed + 1},
                                   solver=pymzn.opturion)[0]
            self._improves[_frx] = _impr
        return self._improves[_frx]


class CoactiveModel:

    def __init__(self, domain):
        self.domain = domain
        self.w = np.random.normal(size=(domain.num_features,))

    def phi(self, x, y):
        return self.domain.phi(x, y)

    def infer(self, x):
        return self.domain.infer(x, self.w)

    def improve(self, x, phi, changed):
        return self.domain.improve(x, phi, changed)

    def update(self, x, y, y_bar):
        self.w += self.phi(x, y_bar) - self.phi(x, y)

    def phi_update(self, phi, phi_bar):
        self.w += phi_bar - phi

