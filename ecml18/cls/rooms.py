import os

from cls.utils import freeze,subdict
from cls.domain import Domain
from sklearn.utils import check_random_state

import numpy as np
import pymzn


class Rooms(object):

    _phi_keys = ["x","y","dx","dy","side_diff"]

    _inference_vars = _phi_keys + ["belong_to","utility"]
    _phi_vars = ["phi","all_normalizers"]
    def __init__(self, seed=None, num_contexts=100, n_rooms=4, **kwargs):
        n_rooms = int(n_rooms)
        self.contexts = self._generate_contexts(num_contexts=num_contexts,n_rooms=n_rooms,seed=seed)
        self.num_features = 45 # AGGIORNARE CON NUMERO DEFINITIVO
        self._phis = {}

        self.features = []

    @property
    def phi_file(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        return directory + "/rooms/phi.mzn"

    @property
    def inference_file(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        return directory + "/rooms/inference.mzn"

    def phi(self, x, y, features=None):
        if features is None:
            features = self.features

        _frx = (freeze(x), freeze(y), tuple(sorted(features)))
        if _frx in self._phis:
            return self._phis[_frx]

        sol = pymzn.minizinc(self.phi_file, 
                            data={
                                    **inputize(subdict(y,keys=Rooms._phi_keys),Rooms._phi_keys),
                                    **x},
                            output_vars=Rooms._phi_vars,
                            solver=pymzn.opturion,
                            suppress_segfault=True
                )
        _phi = [p*n for p,n in zip (sol[-1]['phi'],sol[-1]['all_normalizers'])]
        self._phis[_frx] = np.array(_phi, dtype=np.float64)
        return self._phis[_frx]


    def infer(self, x, w, features=None,timeout=600):
        results =  pymzn.minizinc(self.inference_file, 
                                data={**x,"w":w}, 
                                timeout=timeout, 
                                output_vars=Rooms._inference_vars, 
                                #parse_output=False,
                                solver=pymzn.opturion,
                                suppress_segfault=True
                                )

        return results[-1]

    def _generate_contexts(self, num_contexts=100, n_rooms=4,  seed=None):
        rng = check_random_state(seed)
        #np.random.seed(seed)
        buildings = [
                 {
            "SIDE" : 10,
            "APARTMENT_AREA" : 10*10 - 3*3,
            "N_WALL" : 1,
            "MINIMUM_AREA_PER_ROOM" : 0.05,
            "NUM_MAX_RATIO" : 1,
            "DEN_MAX_RATIO" : 3,
            "door_x" : 1,
            "door_y" : 5,
            "wall_x": [1],
            "wall_y" : [1],
            "wall_dx" : [3],
            "wall_dy" : [3]
                },
                 {
            "SIDE" : 10,
            "APARTMENT_AREA" : 10*10 - 2*4 - 3*4,
            "N_WALL" : 2,
            "MINIMUM_AREA_PER_ROOM" : 0.05,
            "NUM_MAX_RATIO" : 1,
            "DEN_MAX_RATIO" : 3,
            "door_x" : 10,
            "door_y" : 6,
            "wall_x": [9,8],
            "wall_y" : [1,7],
            "wall_dx" : [2,3],
            "wall_dy" : [4,4]
                },
                 {
            "SIDE" : 10,
            "APARTMENT_AREA" : 10*10 - 4*4,
            "N_WALL" : 1,
            "MINIMUM_AREA_PER_ROOM" : 0.05,
            "NUM_MAX_RATIO" : 1,
            "DEN_MAX_RATIO" : 3,
            "door_x" : 5,
            "door_y" : 1,
            "wall_x": [7],
            "wall_y" : [4],
            "wall_dx" : [4],
            "wall_dy" : [4]
                },
                 {
            "SIDE" : 10,
            "APARTMENT_AREA" : 10*10 - 4*4 - 2*4,
            "N_WALL" : 2,
            "MINIMUM_AREA_PER_ROOM" : 0.05,
            "NUM_MAX_RATIO" : 1,
            "DEN_MAX_RATIO" : 3,
            "door_x" : 5,
            "door_y" : 2,
            "wall_x": [7,1],
            "wall_y" : [9,1],
            "wall_dx" : [4,4],
            "wall_dy" : [2,4]
                }
                ]

        def pick_random_rooms(n_rooms):
            l = []
            while sum(l) < n_rooms:
                n = rng.randint(1,3)
                if sum(l) + n <= n_rooms : l += [n]

            if len(l) < 6 : l += [0]*(6-len(l))
            rng.shuffle(l)
            return l

        ctx_list = []
        for i in range(num_contexts):
            rt_ub = pick_random_rooms(n_rooms)
            rt_lb = [rng.randint(i+1) for i in rt_ub]
            
            building = rng.choice(buildings)

            context = {
                    "rt_ub" : rt_ub,
                    "rt_lb" : rt_lb,
                    "SUB_X_ROOM" : 1,
                    **building
                    }
            ctx_list += [context]

        ctx = {
            "SIDE" : 10,
            "APARTMENT_AREA" : 10*10 - 3*3,
            "SUB_X_ROOM" : 1,
            "N_WALL" : 1,
            "MINIMUM_AREA_PER_ROOM" : 0.05,
            "NUM_MAX_RATIO" : 1,
            "DEN_MAX_RATIO" : 3,
            "rt_lb" : [0,0,0,0,0,0],
            "rt_ub" : [1,1,0,0,0,1],
            "door_x" : 1,
            "door_y" : 5,
            "wall_x": [1],
            "wall_y" : [1],
            "wall_dx" : [3],
            "wall_dy" : [3]
                } 
        return ctx_list

    def draw_context(self, user, it):
        return self.contexts[it]

def inputize(x,target_keys):
    x_input = { key:val for key,val in x.items() if key not in target_keys}
    for k in target_keys:
        x_input["input_"+k] = x[k]

    return x_input
