#!/usr/bin/env python3

import re
import shelve
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

marks = ['s-', 'D-', '<-', '^-', '>-', 'v-']
#colors = ['#33A02C', '#1F78B4', '#E31A1C', '#065143', '#f46036']
#fill_colors = ['#B2DF8A', '#A6CEE3', '#FB9A99', '#498076', '#f78b6c']

colors = ['#cc0000', '#3465a4', '#73d216', '#75507b', '#f57900', '#c17d11', '#edd400']
fill_colors = ['#ef2929', '#729fcf', '#8ae234', '#ad7fa8', '#fcaf3e', '#e9b96e', '#fce94f']

def get_label(input_file_group):
    input_file = input_file_group[0]
    if 'rooms' in input_file:
        label = 'CL 5 rooms'
    elif 'tables' in input_file:
        if r'/6/' in input_file:
            label = 'CL 6 tables'
        elif r'/8/' in input_file:
            label = 'CL 8 tables'
        elif r'/10/' in input_file:
            label = 'CL 10 tables'
    if 'approx' in input_file:
        label += ' (approx'
        if 't5' in input_file:
            label += ' 5s'
        elif 't10' in input_file:
            label += ' 10s'
        elif 't20' in input_file:
            label += ' 20s'
        label += ')'
    return label

def label_from_filename(path):
    return path.split("/")[-1].partition(".")[0]


def regret_matrix(traces, iters):
    return np.array([[trace[i][0] if i < len(trace) else 0.0
                      for i in range(iters)] for trace in traces])


def avg_regret_matrix(traces, iters):
    avg_regs = []
    for trace in traces:
        avg_reg = []
        for i in range(iters):
            if i == 0:
                avg_reg.append(trace[i][0])
            elif i < len(trace):
                reg_t = (avg_reg[-1] + trace[i][0])
                avg_reg.append(reg_t)
            else:
                avg_reg.append(avg_reg[-1])
        for i in range(iters):
            avg_reg[i] /= i + 1
        avg_reg.append(avg_reg[-1])
        avg_regs.append(avg_reg)
    return np.array(avg_regs)



def time_matrix(traces, iters):
    times = []
    for trace in traces:
        time = [0.0]
        for i in range(iters):
            if i < len(trace):
                time.append(time[-1] + trace[i][1])
            else:
                time.append(time[-1])
        times.append(time)
    return np.array(times)


def plot_reg(input_files, output_file, iters, no_std=True):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Average regret')

    ax.set_xbound((1, iters))
    x = np.arange(iters + 1)

    y_max = 0

    for i, input_file_group in enumerate(input_files):
        traces = []
        for input_file in input_file_group:
            with shelve.open(input_file) as shelf:
                if 's4' in input_file:
                    traces += list(shelf['traces'].values())[0:4]
                else:
                    traces += shelf['traces'].values()
        label = get_label(input_file_group)

        reg = avg_regret_matrix(traces, iters)
        median = np.median(reg, axis=0)
        std = np.std(reg, axis=0) / np.sqrt(reg.shape[0])

        y_max = max([y_max, median.max()])

        ax.plot(x, median, marks[i], linewidth=1.2, color=colors[i],
                label=label, markevery=4, markersize=4)
        if not no_std:
            ax.fill_between(x, median - std, median + std,
                            color=fill_colors[i], alpha=0.55, linewidth=0)

    #y_max = 0.5
    ax.set_ybound((0, y_max))
    ax.set_yticks(np.arange(0, y_max, 0.2))
    ax.set_ylim([0.0, y_max])
    ax.legend()

    fig.savefig(output_file + '_reg.png', dpi=300, bbox_inches='tight')


def plot_time(input_files, output_file, iters, no_std=True):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cumulative time (minutes)')

    ax.set_xbound((0, iters + 1))
    x = np.arange(iters + 1)

    y_max = 0

    for i, input_file_group in enumerate(input_files):
        traces = []
        for input_file in input_file_group:
            with shelve.open(input_file) as shelf:
                if 's4' in input_file:
                    traces += list(shelf['traces'].values())[0:4]
                else:
                    traces += shelf['traces'].values()
        label = get_label(input_file_group)

        time = time_matrix(traces, iters)
        median = np.median(time, axis=0) / 60
        std = np.std(time, axis=0) / np.sqrt(time.shape[0])

        y_max = max([y_max, median.max()])

        ax.plot(x, median, marks[i], linewidth=1.2, color=colors[i],
                label=label, markevery=4, markersize=4)
        if not no_std:
            ax.fill_between(x, median - std, median + std,
                            color=fill_colors[i], alpha=0.55, linewidth=0)

    ax.set_ybound((0, y_max))
    ax.set_yticks(np.arange(0, y_max + 5, 10))
    ax.set_ylim([0.0, y_max + 5])
    ax.legend(loc="upper left")

    fig.savefig(output_file + '_time.png', dpi=300, bbox_inches='tight')


def aggregate(files):
    curr_file = None
    curr_group = []
    groups = []
    for f in files:
        if not curr_file:
            curr_file = f
            p = re.sub('_s[0-9]_', '_s[0-9]_', curr_file)
            curr_group.append(curr_file)
        else:
            m = re.match(p, f)
            if m:
                curr_group.append(f)
            else:
                curr_file = f
                p = re.sub('_s[0-9]_', '_s[0-9]_', curr_file)
                groups.append(curr_group)
                curr_group = [curr_file]
    groups.append(curr_group)
    return groups


if __name__ == '__main__':
    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)

    parser.add_argument('input_files', nargs='+',
                        help='The pickle files of the results to plot')
    parser.add_argument('-O', '--output-file', default='plot',
                        help='The output file name')
    parser.add_argument('--no-std', action='store_true',
                        help='Whether to plot the standard deviation')
    parser.add_argument('-T', '--iters', type=int, default=100,
                        help='The iterations of to plot')
    args = parser.parse_args()

    files = aggregate(args.input_files)

    plot_reg(files, args.output_file, args.iters, no_std=args.no_std)
    plot_time(files, args.output_file, args.iters, no_std=args.no_std)

