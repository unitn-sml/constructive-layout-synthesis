import os

from cls.utils import freeze,subdict
from cls.domain import Domain
from sklearn.utils import check_random_state

import numpy as np
import pymzn


class Tables(object):

    _phi_keys = ["x","y","dx","dy"]

    _inference_vars = _phi_keys + ["utility"]
    _phi_vars = ["phi","normalizers"]
    def __init__(self, seed=None, num_contexts=100,canvas_size=12, n_tables=4, **kwargs):
        n_tables = int(n_tables)
        self.contexts = self._generate_contexts(num_contexts=num_contexts,
                                                n_tables=n_tables,
                                                canvas_size=canvas_size,
                                                seed=seed)
        self.num_features = 10 # AGGIORNARE CON NUMERO DEFINITIVO
        self._phis = {}

        self.features = []

    @property
    def phi_file(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        return directory + "/tables/phi.mzn"

    @property
    def inference_file(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        return directory + "/tables/infer.mzn"

    def phi(self, x, y, features=None):
        if features is None:
            features = self.features

        _frx = (freeze(x), freeze(y), tuple(sorted(features)))
        if _frx in self._phis:
            return self._phis[_frx]

        sol = pymzn.minizinc(self.phi_file, 
                            data={
                                    **inputize(subdict(y,keys=Tables._phi_keys),Tables._phi_keys),
                                    **x},
                            output_vars=Tables._phi_vars,
                            solver=pymzn.opturion,
                            suppress_segfault=True
                )
        _phi = [p*n for p,n in zip (sol[-1]['phi'],sol[-1]['normalizers'])]
        self._phis[_frx] = np.array(_phi, dtype=np.float64)
        return self._phis[_frx]


    def infer(self, x, w, features=None,timeout=600):
        results =  pymzn.minizinc(self.inference_file, 
                                data={**x,"w":w}, 
                                timeout=timeout, 
                                output_vars=Tables._inference_vars, 
                                #parse_output=False,
                                solver=pymzn.opturion,
                                suppress_segfault=True
                                )

        return results[-1]

    def _generate_contexts(self, num_contexts=100, n_tables=4, canvas_size=12,  seed=None):
        rng = check_random_state(seed)
        #np.random.seed(seed)
        

        rooms = [
                 {
            "SIDE" : canvas_size,
            "N_TABLES" : n_tables,
            "N_WALLS" : 1,
            "door_x" : [1,10],
            "door_y" : [5,5],
            "wall_x": [1],
            "wall_y" : [1],
            "wall_dx" : [3],
            "wall_dy" : [3]
                },
                 {
            "SIDE" : canvas_size,
            "N_TABLES" : n_tables,
            "N_WALLS" : 2,
            "door_x" : [1,10],
            "door_y" : [6,6],
            "wall_x": [9,8],
            "wall_y" : [1,7],
            "wall_dx" : [2,3],
            "wall_dy" : [4,4]
                },
                 {
            "SIDE" : canvas_size, # qualitative cafe setting
            "N_TABLES" : n_tables,
            "N_WALLS" : 1,
            "door_x" : [1,10],
            "door_y" : [4,12],
            "wall_x": [1],
            "wall_y" : [5],
            "wall_dx" : [6],
            "wall_dy" : [8]
                },
                 {
            "SIDE" : canvas_size,
            "N_TABLES" : n_tables,
            "N_WALLS" : 2,
            "door_x" : [5,1],
            "door_y" : [1,8],
            "wall_x": [7,1],
            "wall_y" : [9,1],
            "wall_dx" : [4,4],
            "wall_dy" : [2,4]
                },
                 {
            "SIDE" : canvas_size,
            "N_TABLES" : n_tables,
            'door_x': [7, 4], # qualitative office setting
            'door_y': [1, canvas_size],
            'N_WALLS': 2,
            'wall_x': [1, 8],
            'wall_y': [1, canvas_size - 2],
            'wall_dx': [5, 5],
            'wall_dy': [6, 3]
                }
                ]

        ctx_list = rng.choice(rooms,size=num_contexts)
        return ctx_list

    def draw_context(self, user, it):
        return self.contexts[it]

def inputize(x,target_keys):
    x_input = { key:val for key,val in x.items() if key not in target_keys}
    for k in target_keys:
        x_input["input_"+k] = x[k]

    return x_input
