#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from plot_coincidences import coincidences, add_voltage

dt = (-1,0,1)

def get_eff(a, b, c, center, verbose=False): 

    ab  = coincidences(a,b)
    bc  = coincidences(b,c)
    ca  = coincidences(c,a)

    if center == 'a': 
        abc = np.unique(np.hstack([np.intersect1d(bc,a+t) for t in dt]))
        denom = bc.size
    elif center == 'b':
        abc = np.unique(np.hstack([np.intersect1d(ca,b+t) for t in dt]))
        denom = ca.size
    elif center == 'c':
        abc = np.unique(np.hstack([np.intersect1d(ab,c+t) for t in dt]))
        denom = ab.size

    eff = abc.size / denom
    eff_err = np.sqrt(eff * (1-eff) / denom)

    if verbose:
        print("a:   ", a.size)
        print("b:   ", b.size)
        print("c:   ", c.size)
        print("ab:  ", ab.size)
        print("bc:  ", bc.size)
        print("ca:  ", ca.size)
        print("abc: ", abc.size)
        print()
        print("2pt noise: ", a.size * c.size * len(dt) / (a.max() - a.min()))
        print()

        print("eff = {:.5f} +/- {:.5f}".format(eff, eff_err))
 
    return eff, eff_err, abc.size


def get_eff_curve(timestamps, thresholds, center, thresh_range=(0,256), verbose=False):
    a, b, c = timestamps
    thresh_a, thresh_b, thresh_c = thresholds

    if center.lower() == 'a':
        center_millis = a
        center_thresh = thresh_a
    elif center.lower() == 'b':
        center_millis = b
        center_thresh = thresh_b
    elif center.lower() == 'c':
        center_millis = c
        center_thresh = thresh_c
    else:
        raise ValueError('Invalid center selection')

    thresh = np.unique(center_thresh)

    thresh_min, thresh_max = thresh_range
    thresh = thresh[(thresh >= thresh_min) & (thresh < thresh_max)]

    eff_all = []
    eff_err = []
    rates_all = []
    rates_err = []

    for t in thresh:
        if verbose:
            print('Threshold:', t)
        t_millis = center_millis[center_thresh == t]
        dt_millis = np.diff(t_millis)
        dt_max = 1800000 # 30 min

        tmin = [t_millis.min()] + list(t_millis[1:][dt_millis > dt_max])
        tmax = list(t_millis[:-1][dt_millis > dt_max]) + [t_millis.max()]

        tmin = np.array([tmin]).transpose()
        tmax = np.array([tmax]).transpose()

        thresh_cut_a = np.any((a > tmin) & (a < tmax), axis=0)
        thresh_cut_b = np.any((b > tmin) & (b < tmax), axis=0)
        thresh_cut_c = np.any((c > tmin) & (c < tmax), axis=0)

        a_cut = a[thresh_cut_a]
        b_cut = b[thresh_cut_b]
        c_cut = c[thresh_cut_c]

        eff, err, tot = get_eff(a_cut, b_cut, c_cut, center, verbose=verbose)
        eff_all.append(eff)
        eff_err.append(err)

        if center == 'a':
            counts = a_cut.size
        elif center == 'b':
            counts = b_cut.size
        elif center == 'c':
            counts = c_cut.size

        duration = np.sum(tmax - tmin) / 1000

        if verbose:
            print("tot time:  {:.2f} hr".format(duration / 3600))
            print("abc rate:  {:.5f} mHz".format(1000* tot / duration))
            print('-----')

        rates_all.append((counts - 2*tmin.size) / duration)
        rates_err.append(np.sqrt(counts - 2*tmin.size) / duration)

    return tuple(map(np.array, (thresh, eff_all, eff_err, rates_all, rates_err)))

    
if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('infile', metavar='FILE', help='file to process')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-a', action='store_true')
    group.add_argument('-b', action='store_true')
    group.add_argument('-c', action='store_true')
    parser.add_argument('--tmin', type=float, default=0, help='Minimum millistamp time')
    parser.add_argument('--tmax', type=float, default=1e20, help='Maximum millistamp time')
    parser.add_argument('--thresh_range', type=int, nargs=2, default=(0,256), help='Min and max thresholds')
    parser.add_argument('-t', '--thresh', action='store_true', help='Graph efficiency dependence on threshold')
    parser.add_argument('-T', '--thresh_verbose', action='store_true', help='Graph and print counts')

    args = parser.parse_args()

    try:
        npz = np.load(args.infile)
    except:
        print("could not process file ", args.infile, " as .npz file.")
        exit(1)

    a = npz['millis_a']
    b = npz['millis_b']
    c = npz['millis_c']

    thresh_a = npz['thresh_a']
    thresh_b = npz['thresh_b']
    thresh_c = npz['thresh_c']

    # apply time cuts
    t_cut_a = (a >= args.tmin) & (a < args.tmax)
    t_cut_b = (b >= args.tmin) & (b < args.tmax)
    t_cut_c = (c >= args.tmin) & (c < args.tmax)

    a = a[t_cut_a]
    b = b[t_cut_b]
    c = c[t_cut_c]

    thresh_a = thresh_a[t_cut_a]
    thresh_b = thresh_b[t_cut_b]
    thresh_c = thresh_c[t_cut_c]


    if args.a:
        center = 'a'
    elif args.b:
        center = 'b'
    elif args.c:
        center = 'c'

    if args.thresh or args.thresh_verbose:

        thresh, eff, eff_err, rates, rates_err = get_eff_curve((a,b,c), 
                (thresh_a,thresh_b,thresh_c), 
                center, 
                args.thresh_range, 
                args.thresh_verbose)

        plt.figure(figsize=(5,3))
        ax = plt.gca()
        ax.errorbar(thresh, eff, yerr=eff_err, c='b', marker='o', ms=3)
        
        ax2 = ax.twinx()
        ax2.errorbar(thresh, rates, yerr=rates_err, c='r', marker='o', ms=3)

        ax.set_xlabel('PWM threshold: {}'.format(center))
        ax.set_ylabel(r'$\epsilon$', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_ylim(bottom=0)
        if args.thresh_range[0] > 0:
            ax.set_xlim(left = thresh_min-1)
        if args.thresh_range[1] < 256:
            ax.set_xlim(right = thresh_max)

        ax2.set_ylabel('Count rate (Hz)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(0, .055)
        
        add_voltage(ax)

        plt.tight_layout()
        plt.show()

    else:
        get_eff(a, b, c, center, verbose=True)

    npz.close()
