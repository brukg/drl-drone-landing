#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', metavar='CSVFILE', help='input .csv file')
    parser.add_argument('--time', dest='time', action='store_true',
                        help='Plot against time instead of episode')
    args = parser.parse_args()

    try:
        data = np.genfromtxt(args.csvfile, delimiter=',', skip_header=1)
    except Exception:
        print('Unable to open file %s' % args.csvfile)
        exit(1)

    if len(data.shape) < 2:
        print('Just one point: Mean = %+3.3f  Std = %+3.3f  Max = %+3.3f' %
              (data[1], data[2], data[3]))
        exit(0)

    e = data[:, 0]
    t = data[:, 1]
    r = data[:, 2]

    if args.time:
        plt.plot(t, r)
        plt.xlabel('Time (sec)')
    else:
        plt.plot(e, r)
        plt.xlabel('Episode')

    plt.ylabel('Reward')
    plt.title(args.csvfile)

    plt.show()


main()
