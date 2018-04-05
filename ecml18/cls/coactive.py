# -*- encoding: utf-8 -*-

import numpy as np
from time import time
from cls.utils import *
from sklearn.utils import check_random_state
from textwrap import dedent


class PreferencePerceptron(object):

    @staticmethod
    def init_weights(dim,seed=None):
        rng = check_random_state(seed)
        return rng.normal(scale=0.01,size=(dim,))

    def __call__(self, w, *args, **kwargs):
        return self.update(w, *args, **kwargs)

    def update(self, w, phi_y, phi_y_bar, **kwargs):
        """Updates the weight vector.

        Parameters
        ----------
        phi_y : numpy.ndarray
            The feature vector of the argmax y
        phi_y_bar : numpy.ndarray
            The feature vector of the improvement y_bar
        """
        return w + phi_y_bar - phi_y


class CoactiveLearning(object):
    """
    """
    def __init__(self, domain, learner, w=None,seed=None):
        self.domain = domain
        self.learner = learner
        self.num_features = domain.num_features
        self.w = w
        if self.w is None:
            self.w = learner.init_weights(self.num_features,seed)

    def phi(self, x, y):
        return self.domain.phi(x, y)

    def infer(self, x,timeout= 600):
        return self.domain.infer(x, self.w,timeout=timeout)

    def update(self, *args, **kwargs):
        """Updates the learning model with new evidence."""
        self.w = self.learner(self.w, *args, **kwargs)

    def utility(self, x, y):
        return self.w.dot(self.phi(x, y))

    def simulate(self, user_model, max_iters=100, stop_on_satisfied=False,
                 timeout=600, **kwargs):
        """Simulate the interaction with the given user response model.

        Parameters
        ----------
        user_model : UserResponseModel
            The user response model to interact with.
        max_iters : int
            The maximum number of iterations to use in the elicitation.
        stop_on_satisfied : bool
            Whether to stop the interaction when the user is satisfied.

        Returns
        -------
        generator of tuples
            A generator of tuples with the trace of the algorithm.
        """
        user = user_model
        log = get_logger(__name__)
        log.push_context('uid = {:>2d}'.format(user.uid))

        msg = dedent('''
            uid = {user.uid:>2d}, it = {it:>2d}, reg = {reg:>7.3f}, t = {t:>7.3f}
        ''').strip()

        data = []
        for it in range(max_iters):
            log.push_context('it = {:>2d}'.format(it))

            # Receive context
            x = user.draw_context(it)
            log.debug('x = {x}', locals())

            # Inference
            t0 = time()
            y = self.infer(x, timeout=timeout)
            t_infer = time() - t0
            log.debug('''
                t_infer = {t_infer}
                y = {y}
            ''', locals())

            reg = user.regret(x, y)
            if stop_on_satisfied and user.satisfied(x, y):
                log.pop_context()
                log.debug('''
                    user satisfied
                ''', locals())
                t = t_infer
                print(msg.format(**locals()))
                yield reg, t
                break

            # Improvement
            t1 = time()
            y_bar = user.improve(x, y)
            t_improve = time() - t1
            log.debug('''
                t_improve = {t_improve}
                y_bar     = {y_bar}
            ''', locals())

            # Model update
            w0 = self.w.copy()
            t2 = time()
            phi_y = self.phi(x, y)
            phi_y_bar = self.phi(x, y_bar)
            self.update(phi_y, phi_y_bar)
            t_update = time() - t2
            w1 = self.w

            log.debug('''
                t_update  = {t_update}
                w0        = {w0}
                phi_y     = {phi_y}
                phi_y_bar = {phi_y_bar}
                w1        = {w1}
            ''', locals())

            t = t_infer + t_update

            print(msg.format(**locals()))
            yield reg, t

            log.pop_context()
        else:
            log.debug('''
                user not satisfied
            ''', locals())
        log.pop_context()

