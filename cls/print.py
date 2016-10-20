#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import pymzn


def draw(args):

    size = args['size']

    with open(args['dzn']) as f:
        dzn = pymzn.parse_dzn(f.read())

    x = dzn['x']
    y = dzn['y']
    dx = dzn['dx']
    dy = dzn['dy']

    assert(len(x) == len(dx) == len(y) == len(dy))
    assert(all([x[i] + dx[i] <= size and
                y[i] + dy[i] <= size
                for i in range(len(x))]))

    m = [[' · ' for __ in range(size)] for __ in range(size)]

    for i in range(len(x)):
        x_i = x[i]
        dx_i = dx[i]
        y_i = y[i]
        dy_i = dy[i]
        for j in range(y_i, y_i + dy_i):
            for k in range(x_i, x_i + dx_i):
                if dx_i == 1:
                    m[size - j][k - 1] = '[+]'
                elif k == x_i:
                    m[size - j][k - 1] = '[++'
                elif k == x_i + dx_i - 1:
                    m[size - j][k - 1] = '++]'
                else:
                    m[size - j][k - 1] = '+++'

    print(' ' + '---' * size + ' ')
    print('|' + '|\n|'.join([''.join(l) for l in m]) + '|')
    print(' ' + '---' * size + ' ')


if __name__ == '__main__':
    import argparse

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)

    parser.add_argument("dzn", help="The input in dzn format")
    parser.add_argument("--size", type=int, default=100,
                        help="The size of the canvas")
    args = parser.parse_args()

    draw(vars(args))

