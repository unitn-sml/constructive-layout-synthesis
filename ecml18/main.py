#!/bin/python

import sys
import pickle
import shelve
import logging
import argparse
import threading
import os.path

from sklearn.utils import check_random_state

from cls.utils import *
from cls.coactive import PreferencePerceptron
from cls.coactive import CoactiveLearning
from cls.users import User, sample_users, RoomsCoactiveFeedback, TablesCoactiveFeedback
from cls.rooms import Rooms
from cls.tables import Tables
from itertools import combinations


LEARNERS = {
    'pp': PreferencePerceptron,
}


def generate_domain(**kwargs):
    klass = get_class(kwargs['domain'], 'cls')
    args = get_defaults(klass.__init__)
    args = {**args, **subdict(kwargs, nokeys=['args'])}
    if 'args' in kwargs and kwargs['args']:
        if isinstance(kwargs['args'], list):
            args = {**args, **parse_remainder(kwargs['args'])}
        elif isinstance(kwargs['args'], dict):
            args = {**args, **kwargs['args']}
        else:
            raise ValueError('invalid args')
    domain = klass(**args)

    if 'domain_shelf' in kwargs and kwargs['domain_shelf']:
        with shelve.open(kwargs['domain_shelf']) as shelf:
            shelf['args'] = args
            shelf['domain'] = domain
    return domain


def generate_users(**kwargs):
    if 'domain' in kwargs and kwargs['domain']:
        domkeys = ['domain', 'domain_shelf', 'args']
        domain = generate_domain(**subdict(kwargs, keys=domkeys))
    elif 'domain_shelf' in kwargs and kwargs['domain_shelf']:
        with shelve.open(kwargs['domain_shelf']) as shelf:
            domain = shelf['domain']
    else:
        raise ValueError('no domain specified')

    args = subdict(kwargs, nokeys={'domain'})
    users = sample_users(domain, **args)

    with shelve.open(kwargs['users_shelf']) as shelf:
        shelf['args'] = args
        shelf['domain'] = domain
        shelf['users'] = users

    return users


def simulate(**kwargs):
    with shelve.open(kwargs['user_shelf']) as shelf:
        domain = shelf['domain']
        users = shelf['users']
    if 'domain_shelf' in kwargs and kwargs['domain_shelf']:
        with shelve.open(kwargs['domain_shelf']) as shelf:
            domain = shelf['domain']

    if not kwargs['output_shelf']:
        import os.path
        usrfile, ext = os.path.splitext(kwargs['user_shelf'])
        outfile =  usrfile + '_trace' + ext
    else:
        outfile = kwargs['output_shelf']

    u = int(kwargs['user'])
    n = int(kwargs['num_users']) or len(users)

    for user in users[u:n]:
        learner = LEARNERS[kwargs['learner']]()
        if isinstance(domain, Rooms):
            user_model = RoomsCoactiveFeedback(domain, user)
        else:
            user_model = TablesCoactiveFeedback(domain, user)
        model = CoactiveLearning(domain, learner,seed=kwargs["seed"])
        for t in model.simulate(user_model, **kwargs):
            with shelve.open(outfile, writeback=True) as outshelf:
                if 'traces' not in outshelf:
                    outshelf['traces'] = {}
                if user.uid not in outshelf['traces']:
                    outshelf['traces'][user.uid] = []
                outshelf['traces'][user.uid].append(t)


if __name__ == '__main__':
    fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt)
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='the seed for RNG')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='display logs on standard output')
    parser.add_argument('--log', default='srai.log', help='Log file')

    subparsers = parser.add_subparsers()

    ## GENERATE
    gen_parser = subparsers.add_parser('generate')
    gen_subparsers = gen_parser.add_subparsers()

    gen_users_parser = gen_subparsers.add_parser(
        'users', formatter_class=fmt,
        help='Generate users'
    )
    gen_users_parser.set_defaults(cmd=generate_users)
    gen_users_parser.add_argument(
        '-U', '--users-shelf', required=True,
        help='the file where to store the users'
    )
    gen_users_parser.add_argument(
        '-d', '--domain',
        help='the domain class to generate'
    )
    gen_users_parser.add_argument(
        '-D', '--domain-shelf',
        help='the file containing the domain'
    )
    gen_users_parser.add_argument(
        '-n', '--num-users', type=int, default=20,
        help='the number of users to generate'
    )
    gen_users_parser.add_argument(
        '--dist-w', choices={'normal', 'uniform'}, default='normal',
        help='the distribution of the user weights'
    )
    gen_users_parser.add_argument(
        '--dist-phi', choices={'none', 'base', 'sample'}, default='none',
        help='the distribution of the user features'
    )
    gen_users_parser.add_argument(
        '--non-negative', action='store_true',
        help='whether to generate only non negative user weights'
    )
    gen_users_parser.add_argument(
        '--density', type=float, default=1.0,
        help='the density of the weight vectors (sparsify with density < 1.0)'
    )
    gen_users_parser.add_argument(
        '--mean-dim', type=int, default=10,
        help='the mean dimension of the feature vectors'
    )
    gen_users_parser.add_argument(
        'args', nargs=argparse.REMAINDER,
        help='arguments to pass to the domain'
    )

    gen_domain_parser = gen_subparsers.add_parser(
        'domain', formatter_class=fmt,
        help='Generate a domain'
    )
    gen_domain_parser.set_defaults(cmd=generate_domain)
    gen_domain_parser.add_argument(
        'domain', help='the domain class to generate'
    )
    gen_domain_parser.add_argument(
        '-D', '--domain-shelf', required=True,
        help='the file where to store the domain'
    )
    gen_domain_parser.add_argument(
        'args', nargs=argparse.REMAINDER,
        help='arguments to pass to the domain'
    )

    ## SIMULATE
    simulate_parser = subparsers.add_parser(
        'simulate', formatter_class=fmt,
        help='starts an simulation procedure'
    )
    simulate_parser.set_defaults(cmd=simulate)
    simulate_subparsers = simulate_parser.add_subparsers()

    simulate_parser.add_argument(
        '-U', '--user-shelf',
        help='the file containing the users'
    )
    simulate_parser.add_argument(
        '-D', '--domain-shelf',
        help='the file containing the domain'
    )
    simulate_parser.add_argument(
        '-O', '--output-shelf',
        help='the file where to store the output trace'
    )
    simulate_parser.add_argument(
        '-u', '--user', type=int, default=0,
        help='user to start the simulateation with'
    )
    simulate_parser.add_argument(
        '-n', '--num-users', type=int, default=0,
        help='number of users to use in the simulateation (0 = all)'
    )
    simulate_parser.add_argument(
        '-L', '--learner', choices=list(LEARNERS.keys()), default='pp',
        help='the learning algorithm to use'
    )
    simulate_parser.add_argument(
        '--timeout', type=int, default=600,
        help='timeout for inference'
    )

    coactive_parser = simulate_subparsers.add_parser(
        'coactive', formatter_class=fmt,
        help='starts a coactive learning simulation procedure'
    )
    coactive_parser.set_defaults(model_class=CoactiveLearning)
    coactive_parser.add_argument(
        '--alpha', type=float, default=0.1,
        help='the alpha value of the users'
    )

    args = parser.parse_args()

    handlers = [logging.FileHandler(args.log, mode='w+')]
    if args.verbose:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    if hasattr(args, 'cmd'):
        args.cmd(**vars(args))
    else:
        parser.print_help()

