#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    a = []
    b = []
    c = []

    for fname in args.files:
        f = np.load(fname)
        a.append(f.f.millis_a)
        b.append(f.f.millis_b)
        c.append(f.f.millis_c)

    a = np.hstack(a)
    b = np.hstack(b)
    c = np.hstack(c)

    tmin = np.amin([a.min(), b.min(), c.min()])
    tmax = np.amax([a.max(), b.max(), c.max()])

    bins = np.linspace(tmin, tmax, 100)

    plt.hist(a, bins=bins, histtype='step', label='A')
    plt.hist(b, bins=bins, histtype='step', label='B')
    plt.hist(c, bins=bins, histtype='step', label='C')

    plt.xlabel('t [ms]')

    plt.legend()

    plt.show()
