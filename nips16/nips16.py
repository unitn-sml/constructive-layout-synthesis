#!/usr/bin/env python3

import cls
import sys
import pickle
import logging
import argparse
import threading
import numpy as np

from sklearn.utils import check_random_state

from cls.utils import subdict
from cls.coactive import User, pp
from cls.furniture import Furniture


def make_problem(problem, **kwargs):
    return {'furn': Furniture}[problem](**kwargs)


def gen_weights(args):
    if not args['weights']:
        raise ValueError('Argument weights must be given.')

    rng = check_random_state(args['seed'])
    problem = make_problem(args['problem'], **subdict(args, {'problem'}))

    weights = [(uid, rng.normal(size=(problem.num_features,)))
               for uid in range(1, args['users'] + 1)]

    with open(args['weights'], 'wb') as f:
        pickle.dump(weights, f)


def experiment(args):
    rng = check_random_state(args['seed'])
    problem = make_problem(args['problem'], **subdict(args, {'problem'}))

    if args['weights']:
        with open(args['weights'], 'rb') as f:
            weights = pickle.load(f)
        users = [User(problem, w_star, uid=uid, noise=args['noise'], rng=rng,
                 alpha=args['alpha']) for uid, w_star in weights]
    else:
        users = [User(problem, rng.normal(size=(problem.num_features,)),
                      uid=uid, noise=args['noise'], alpha=args['alpha'],
                      rng=rng) for uid in range(1, args['users'] + 1)]

    approx = args['timeout'] is not None

    traces = []
    start_user = args['user']

    pd = args['parallel']
    pdl = (len(users) - start_user) // pd + 1
    batches = [users[u : u + pdl] for u in range(start_user, len(users), pdl)]

    def _exp(_users):
        for user in _users:
            trace = pp(problem, user, max_iters=args['iters'], approx=approx)
            traces.append(trace)

    threads = []
    for batch in batches:
        t = threading.Thread(target=_exp, args=(batch,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    with open(args['output_file'], 'wb') as f:
        pickle.dump((args['label'], traces), f)


if __name__ == '__main__':
    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)

    parser.add_argument('method', choices=['exp', 'gen'],
                        help='The method to execute')
    parser.add_argument('problem', choices=['furn'], default='furn',
                        help='The problem')
    parser.add_argument('-W', '--weights',
                        help='File with true user weights')
    parser.add_argument('-O', '--output-file', default='trace.pickle',
                        help='File to dump the experiment trace')
    parser.add_argument('-u', '--user', type=int, default=0, 
                        help='User to start from')
    parser.add_argument('-U', '--users', type=int, default=20, 
                        help='Number of users')
    parser.add_argument('-T', '--iters', type=int, default=100,
                        help='Number of iterations')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Seed for the random number generator')
    parser.add_argument('-a', '--alpha', type=float, default=0.2,
                        help=('The alpha for the improvement'
                        '(std of normal noise on w_star)'))
    parser.add_argument('-S', '--canvas_size', type=int, default=12,
                        help=('The canvas size'))
    parser.add_argument('-t', '--num_tables', type=int, default=8,
                        help=('The number of tables'))
    parser.add_argument('-L', '--layout', type=int, default=0,
                        help=('The layout'))
    parser.add_argument('-l', '--label', default='exp',
                        help=('The label for the graphs'))
    parser.add_argument('-n', '--noise', type=float, default=None,
                        help=('Noise for the user improvements '
                        '(std of normal noise on w_star)'))
    parser.add_argument('-p', '--parallel', type=int, default=4,
                        help=('The parallelism degree'))
    parser.add_argument('--timeout', type=int, default=None,
                        help=('The timeout for opturion'))
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging on screen')
    parser.add_argument('--log', default='cls.log', help='Log file')
    args = parser.parse_args()

    handlers = [logging.FileHandler(args.log, mode='w+')]
    if args.debug:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    {'exp': experiment,
     'gen': gen_weights}[args.method](vars(args))
