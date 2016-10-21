# -*- encoding: utf-8 -*-

import logging
import numpy as np

from time import time
from cls.utils import BracesAdapter


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
        return np.zeros(self.num_features)

    def phi(self, x):
        """Returns the feature vector of x."""
        raise NotImplementedError()

    def infer(self, w):
        """Infers optimal object."""
        raise NotImplementedError()

    def improve(self, x, w):
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

    Attributes
    ----------
    x_star : dict
        The optimal configuration.
    u_star : float
        The utility of x_star.
    """

    def __init__(self, problem, w_star):
        if w_star.shape != (problem.num_features,):
            raise ValueError('Mismatching w_star')

        self.problem = problem
        self.w_star = w_star
        self.x_star = self.problem.infer(self.w_star)
        self.u_star = self.problem.utility(self.x_star, self.w_star)

    def regret(self, x):
        """Returns the regret of object x."""
        return self.u_star - self.problem.utility(x, self.w_star)

    def improve(self, x):
        """Computes an improvement for x."""
        return self.problem.improve(x, self.w_star)

    def satisfied(self, x):
        """Returns true if regret of object x is zero."""
        return self.regret(x) == 0.0


def pp(problem, user, max_iters=100, uid=0):
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
    uid : positive int
        Identifier for the current user.

    Returns
    -------
    trace : list of tuples
        List of (loss, time) pairs for all iterations.

    References
    ----------
    .. [1] Shivaswamy and Joachims, *Coactive Learning*, JAIR 53 (2015)
    """
    log = BracesAdapter(logging.getLogger(__name__))

    msg = 'uid = {}, it = {}, t = {}, reg = {}'
    w = problem.init_w()
    trace = []
    for i in range(max_iters):
        log.debug('it = {}, w = {}', i, w)

        # Inference
        t0 = time()
        x = problem.infer(w)
        t_infer = time() - t0
        log.debug('uid = {}, it = {}, x = {}', uid, i, x)
        log.debug('uid = {}, it = {}, t_infer = {}', uid, i, t_infer)

        if user.satisfied(x):
            trace.append((0.0, t_infer))
            log.debug('outcome: user satisfied')
            break

        # Improvement
        t1 = time()
        x_bar = user.improve(x)
        t_improve = time() - t1
        log.debug('uid = {}, it = {}, x_bar = {}', uid, i, x_bar)
        log.debug('uid = {}, it = {}, t_improve = {}', uid, i, t_improve)

        # Model update
        t2 = time()
        w += problem.phi(x_bar) - problem.phi(x)
        t_update = time() - t2
        log.debug('uid = {}, it = {}, t_update = {}', uid, i, t_update)

        regret = user.regret(x)
        t_elapsed = t_infer + t_update
        trace.append((regret, t_elapsed))
        log.debug('uid = {}, it = {}, regret = {}', uid, i, regret)
        log.debug('uid = {}, it = {}, regret = {}', uid, i, t_elapsed)

        print(msg.format(uid, i, t_elapsed, regret))

    else:
        log.debug('outcome: user not satisfied')

    return trace

