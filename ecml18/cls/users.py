import os
import re
import pymzn
import numpy as np

from subprocess import CalledProcessError
from sklearn.utils import check_random_state
from cls.utils import *

class User(object):
    """A user used in a simulation experiment.

    Parameters
    ----------
    w_star : numpy.ndarray
        The true weight vector of the user.
    phi_star : list
        The list of true MiniZinc features used by this user.
    uid : int
        The id of the user.
    """
    def __init__(self, w_star, features=None, uid=0):
        self.w_star = w_star
        self.features = features
        self.uid = uid


def sample_users(domain, num_users=20, dist_w='uniform',
                 non_negative=False, density=1.0, dist_phi='none', mean_dim=10,
                 seed=None, **kwargs):
    """Samples a set of users.

    This method samples a set of users according to the given parameters.
    The set of features exposed by the domain is used for sampling the features.
    The weights are always normalized after sampling.

    Parameters
    ----------
    domain : Domain
        The domain used for interacting with the users.
    num_users : int
        The number of users to sample.
    dist_w : one in {'normal', 'uniform'}
        The distribution for sampling the user from.
    non_negative : bool
        Whether to sample only non-negative weights.
    density : float in (0.0, 1.0]
        The density to use to sparsify the weight vectors.
    dist_phi : one in {'base', 'sample'}
        The method for sampling the features.
    mean_dim : int
        Mean dimension of the feature vector to be sampled. The actual dimension
        is sample from a symmetric binomial distribution with this mean.
    seed : int
        The RNG seed.
    """
    rng = check_random_state(seed)
    users = []
    for uid in range(num_users):
        features = {
            'none': lambda: [],
            'base': lambda: domain.features,
            'sample': lambda: domain.sample_features(
                                 rng.binomial(mean_dim * 2, 0.5), **kwargs)
        }[dist_phi]()
        w_star = {
            'normal' : lambda: rng.normal(0, 1, domain.num_features),
            'uniform': lambda: rng.uniform(-1, 1, domain.num_features)
        }[dist_w]()

        if non_negative:
            w_star = np.abs(w_star)

        if density < 1.0:
            perm = rng.permutation(w_star.shape[0])
            w_star[perm[:round((1.0 - density) * len(features))]] = 0.0

        w_star /= np.linalg.norm(w_star)
        users.append(User(w_star, features, uid))
    return users


class RoomsCoactiveFeedback(object):
    
    _improvement_keys = ["x","y","dx","dy","belong_to"]
    _improvement_vars = ["x","y","dx","dy","belong_to","side_diff","utility"]

    _utility_vars = ["utility"]
    
    def __init__(self, domain, user, alpha=0.1, noise=None, seed=None):
        self.domain = domain
        self.user = user
        self.alpha = alpha
        self.noise = noise
        self.rng = check_random_state(seed)
        self._y_stars = {}
        directory = (os.path.dirname(os.path.realpath(__file__))) 
        self.improvement_file = directory + "/rooms/improvement.mzn"
        self.utility_file = directory + "/rooms/utility.mzn"

    @property
    def uid(self):
        return self.user.uid

    @property
    def w_star(self):
        return self.user.w_star

    @property
    def features(self):
        return self.user.features

    def draw_context(self, it):
        return self.domain.draw_context(self.user, it)

    def phi_star(self, x, y):
        return self.domain.phi(x, y, self.features)

    def utility(self, x, y):
        """The utility function of the user.

        Calculate the utility function of an object y in context x, with respect
        to the user weights and features.

        Parameters
        ----------
        x : dict
            The context x.
        y : dict
            The object y.
        """
        return self.w_star.dot(self.phi_star(x, y))

    def regret(self, x, y):
        _frx = freeze(x)
        if _frx in self._y_stars:
            y_star = self._y_stars[_frx]
        else:
            y_star = self.domain.infer(x, self.w_star, self.features)
            self._y_stars[_frx] = y_star

        u_y = self.utility(x, y)
        u_star = self.utility(x, y_star)
        reg = u_star - u_y

        log = get_logger(__name__)
        log.debug('''
            y_star = {y_star}
            u_y    = {u_y}
            u_star = {u_star}
            reg    = {reg}
        ''', locals())

        return reg

    def satisfied(self, *args, **kwargs):
        return self.regret(*args, **kwargs) <= 0.0

    def improve(self, x, y, timeout=600):

        frx = freeze(x)
        y_star = None
        if frx in self._y_stars:
            y_star = self._y_stars[freeze(x)]
        else :
            y_star = self.domain.infer(x,self.w_star)

        y_util_data = {
                "input_x" : y["x"],
                "input_y" : y["y"],
                "input_dx" : y["dx"],
                "input_dy" : y["dy"],
                "input_side_diff" : y["side_diff"],
                "w" : self.w_star,
                **x
                }

        y_util = pymzn.minizinc(self.utility_file, 
                            data=y_util_data,
                            output_vars=["utility"],
                                timeout=timeout, 
                            solver=pymzn.opturion,
                            suppress_segfault=True
                )[-1]["utility"]

        improve_data = {
                    "input_belong_to": y["belong_to"],
                    "input_utility" : y_util,
                    "input_star_utility" : y_star["utility"],
                    "alpha" : self.alpha,
                    "w" : self.w_star,
                     **x}
        sol = pymzn.minizinc(self.improvement_file, 
                            data=improve_data,
                            output_vars=RoomsCoactiveFeedback._improvement_vars,
                                timeout=timeout, 
                            solver=pymzn.opturion,
                            suppress_segfault=True
                )[-1]

        #def u(y):
        #    phis = self.domain.phi(x,y)
        #    return sum([p*w for p,w in zip(phis,self.w_star)])
        #print("y: ",y)
        #print("y_util: ",y_util)
        #print("y_real_util: ",u(y))
        #print("y_star_real_util: ",u(y_star))
        #print("y_star: ",y_star)
        #print("y_bar: ",sol)
        return sol

class TablesCoactiveFeedback(object):
    # TODO

    _improvement_keys = ["x","y","dx","dy"]
    _improvement_vars = ["x","y","dx","dy"]

    _utility_vars = ["utility"]
    
    def __init__(self, domain, user, alpha=0.1, noise=None, seed=None):
        self.domain = domain
        self.user = user
        self.alpha = alpha
        self.noise = noise
        self.rng = check_random_state(seed)
        self._y_stars = {}
        directory = (os.path.dirname(os.path.realpath(__file__))) 
        self.improvement_file = directory + "/tables/improve.mzn"
        self.utility_file = directory + "/tables/utility.mzn"

    @property
    def uid(self):
        return self.user.uid

    @property
    def w_star(self):
        return self.user.w_star

    @property
    def features(self):
        return self.user.features

    def draw_context(self, it):
        return self.domain.draw_context(self.user, it)

    def phi_star(self, x, y):
        return self.domain.phi(x, y, self.features)

    def utility(self, x, y):
        """The utility function of the user.

        Calculate the utility function of an object y in context x, with respect
        to the user weights and features.

        Parameters
        ----------
        x : dict
            The context x.
        y : dict
            The object y.
        """
        return self.w_star.dot(self.phi_star(x, y))

    def regret(self, x, y):
        _frx = freeze(x)
        if _frx in self._y_stars:
            y_star = self._y_stars[_frx]
        else:
            y_star = self.domain.infer(x, self.w_star, self.features)
            self._y_stars[_frx] = y_star

        u_y = self.utility(x, y)
        u_star = self.utility(x, y_star)
        reg = u_star - u_y

        log = get_logger(__name__)
        log.debug('''
            y_star = {y_star}
            u_y    = {u_y}
            u_star = {u_star}
            reg    = {reg}
        ''', locals())

        return reg

    def satisfied(self, *args, **kwargs):
        return self.regret(*args, **kwargs) <= 0.0

    def improve(self, x, y):

        if self.satisfied(x,y): return y

        frx = freeze(x)
        y_star = None
        if frx in self._y_stars:
            y_star = self._y_stars[freeze(x)]
        else :
            y_star = self.domain.infer(x,self.w_star)


        improve_data = {
                    "ALPHA" : self.alpha,
                    "input_x" : y["x"],
                    "input_y" : y["y"],
                    "input_dx" : y["dx"],
                    "input_dy" : y["dy"],
                    "input_star_x" : y_star["x"],
                    "input_star_y" : y_star["y"],
                    "input_star_dx" : y_star["dx"],
                    "input_star_dy" : y_star["dy"],
                    "w" : self.w_star,
                     **x}
        sol = pymzn.minizinc(self.improvement_file, 
                            data=improve_data,
                            output_vars=TablesCoactiveFeedback._improvement_vars,
                            solver=pymzn.opturion,
                            suppress_segfault=True
                )[-1]

        #def u(y):
        #    phis = self.domain.phi(x,y)
        #    return sum([p*w for p,w in zip(phis,self.w_star)])
        #print("y: ",y)
        #print("y_util: ",y_util)
        #print("y_real_util: ",u(y))
        #print("y_star_real_util: ",u(y_star))
        #print("y_star: ",y_star)
        #print("y_bar: ",sol)
        return sol












