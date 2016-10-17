#!/usr/bin/env python3

import cls
import sys
import logging


def elicit(args):
    pass


if __name__ == '__main__':
    import argparse

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)

    parser.add_argument("method", help="The method to execute")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Seed for the random number generator")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging on screen")
    args = parser.parse_args()

    handlers = [logging.FileHandler('cls.log', mode='w+')]
    if args.debug:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    {'elicit': elicit}[args.method](vars(args))
