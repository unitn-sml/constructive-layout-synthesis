# -*- encoding: utf-8 -*-

import numpy as np

from time import time
from cls.utils import get_logger, array2str, x2str
from functools import partial

class Problem(object):
    """Base class for all problems.

    Parameters
    ----------
    num_features : positive int
        The number of features in :math:`\phi`

    Notes
    -----
    The problem includes the methods to compute the feature vector
    :math:`\phi(\mathtt{x})` and make inference:

    .. math:: \text{argmax}_{x \in \mathcal{X}} \ \langle \mathbb{w}, \phi(\mathtt{x}) \rangle
    """
    def __init__(self, num_features):
        self.num_features = num_features

    def init_w(self):
        """Initializes a new feature vector of shape (self.num_features,)."""
        return np.ones(self.num_features)

    def phi(self, x):
        """Returns the feature vector of x."""
        raise NotImplementedError()

    def infer(self, w):
        """Infers optimal object."""
        raise NotImplementedError()

    def improve(self, x, x_star, w, alpha=0.1):
       """Makes a minimal improvement to x w.r.t. w."""
       raise NotImplementedError()

    def utility(self, x, w):
        """ Computes the utility of x w.r.t. the weights w."""
        return w.dot(self.phi(x))


class User(object):
    """Class for a simulated user with fixed weight vector.
    
    Parameters
    ----------
    problem : Problem
        An instance of Problem.
    w_star : numpy.ndarray of shape (problem.num_features,)
        The true weight vector.
    uid : positive int
        Identifier for the user.

    Attributes
    ----------
    x_star : dict
        The optimal configuration.
    u_star : float
        The utility of x_star.
    """

    def __init__(self, problem, w_star, uid=0, noise=None, rng=None,
                 alpha=0.1):
        if w_star.shape != (problem.num_features,):
            raise ValueError('Mismatching w_star')

        self.problem = problem
        self.w_star = w_star
        self._uid = uid
        self.x_star = None
        self.u_star = None
        self.noise = noise
        self.rng = rng
        self.alpha = alpha

    def init(self):
        self.x_star = self.problem.infer(self.w_star)
        self.u_star = self.problem.utility(self.x_star, self.w_star)
        phi_x_star = self.problem.phi(self.x_star)

        log = get_logger(__name__)
        log.debug('uid = {}, w_star = {}', self._uid,
                  partial(array2str, self.w_star))
        log.debug('uid = {}, x_star = {}', self._uid,
                  partial(x2str, self.x_star))
        log.debug('uid = {}, phi_x_star = {}', self._uid,
                  partial(array2str, phi_x_star))
        log.debug('uid = {}, u_star = {}', self._uid, self.u_star)

    @property
    def uid(self):
        return self._uid

    def utility(self, x):
        """Returns the utility of object x."""
        return self.problem.utility(x, self.w_star)

    def regret(self, x):
        """Returns the regret of object x."""
        return self.u_star - self.utility(x)

    def improve(self, x):
        """Computes an improvement for x."""
        w_star = np.copy(self.w_star)
        if self.noise:
            nnz = w_star.nonzero()[0]
            w_star[nnz] += self.rng.normal(0, self.noise, size=len(nnz))
        return self.problem.improve(x, self.x_star, w_star, self.alpha)


def pp(problem, user, max_iters=100):
    """The Preference Perceptron [1]_.

    This is a context-less implementation, used for preference elicitation.

    Termination occurs when (i) the user is satisfied, or (ii) the maximum
    number of iterations is reached.

    Parameters
    ----------
    problem : Problem
        The target problem.
    user : User
        The user of the perceptron.
    max_iters : positive int
        Number of iterations.

    Returns
    -------
    trace : list of tuples
        List of (loss, time) pairs for all iterations.

    References
    ----------
    .. [1] Shivaswamy and Joachims, *Coactive Learning*, JAIR 53 (2015)
    """
    log = get_logger(__name__)

    user.init()

    msg = 'uid = {}, it = {}, t = {}, reg = {}'
    w = problem.init_w()
    trace = []
    for i in range(max_iters):
        log.debug('uid = {}, it = {}, w = {}', user.uid, i, w)

        # Inference
        t0 = time()
        x = problem.infer(w)
        t_infer = time() - t0
        u_x = user.utility(x)
        phi_x = problem.phi(x)
        log.debug('uid = {}, it = {}, x = {}', user.uid, i,
                  partial(x2str, x))
        log.debug('uid = {}, it = {}, phi_x = {}', user.uid, i,
                  partial(array2str, phi_x))
        log.debug('uid = {}, it = {}, u_x = {}', user.uid, i, u_x)
        log.debug('uid = {}, it = {}, t_infer = {}', user.uid, i, t_infer)

        regret = user.regret(x)
        log.debug('uid = {}, it = {}, regret = {}', user.uid, i, regret)

        if regret == 0.0:
            log.debug('outcome: user satisfied')
            trace.append((regret, t_infer, w.copy()))
            print(msg.format(user.uid, i, t_infer, regret))
            break

        # Improvement
        t1 = time()
        x_bar = user.improve(x)
        t_improve = time() - t1
        phi_x_bar = problem.phi(x_bar)
        u_x_bar = user.utility(x_bar)
        log.debug('uid = {}, it = {}, x_bar = {}', user.uid, i,
                  partial(x2str, x_bar))
        log.debug('uid = {}, it = {}, phi_x_bar = {}', user.uid, i,
                  partial(array2str, phi_x_bar))
        log.debug('uid = {}, it = {}, u_x_bar = {}', user.uid, i, u_x_bar)
        log.debug('uid = {}, it = {}, t_improve = {}', user.uid, i, t_improve)

        # Model update
        t2 = time()
        w += phi_x_bar - phi_x
        t_update = time() - t2
        log.debug('uid = {}, it = {}, t_update = {}', user.uid, i, t_update)

        t_elapsed = t_infer + t_update
        log.debug('uid = {}, it = {}, t_elapsed = {}', user.uid, i, t_elapsed)

        trace.append((regret, t_elapsed, w.copy()))
        print(msg.format(user.uid, i, t_elapsed, regret))

    else:
        log.debug('outcome: user not satisfied')

    return trace

