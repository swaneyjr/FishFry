#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def get_eff(a, b, c, center, t_range=(None, None), verbose=False): 

    tmin, tmax = t_range
    if tmin:
        a = a[a > tmin]
        b = b[b > tmin]
        c = c[c > tmin]
    if tmax:
        a = a[a < tmax]
        b = b[b < tmax]
        c = c[c < tmax]

    dt = (-1,0,1)

    ab  = np.unique(np.hstack([np.intersect1d(a,b+t) for t in dt]))
    bc  = np.unique(np.hstack([np.intersect1d(b,c+t) for t in dt]))
    ca  = np.unique(np.hstack([np.intersect1d(c,a+t) for t in dt]))

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

        print("min time:  ", a.min())
        print("max time:  ", a.max())
        print("tot time:  {:.2f} hr".format((a.max() - a.min()) / 3600000))
        print("abc rate:  {:.5f} Hz".format(1000* abc.size / (a.max() - a.min())))
        print()

    return eff, eff_err

    
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
    parser.add_argument('-t', '--thresh', action='store_true', help='Graph efficiency dependence on threshold')
    parser.add_argument('-T', '--thresh_verbose', action='store_true', help='Graph and print counts')

    args = parser.parse_args()

    try:
        npz = np.load(args.infile)
    except:
        print("could not process file ", args.infile, " as .npz file.")
        exit(1)

    old_format = 'chan_a' in npz.keys()

    a = npz['chan_a'] if old_format else npz['millis_a']
    b = npz['chan_b'] if old_format else npz['millis_b']
    c = npz['chan_c'] if old_format else npz['millis_c']

    if args.a:
        center = 'a' 
    elif args.b:
        center = 'b'
        center_thresh = npz['thresh_b'] if not old_format else None
    elif args.c:
        center = 'c'
        center_thresh = npz['thresh_c'] if not old_format else None
     

    if (args.thresh or args.thresh_verbose) and not old_format:

        if args.a:
            center_millis = npz['millis_a']
            center_thresh = npz['thresh_a']
        elif args.b:
            center_millis = npz['millis_b']
            center_thresh = npz['thresh_b']
        elif args.c:
            center_millis = npz['millis_c']
            center_thresh = npz['thresh_c']

        thresh_unique = np.unique(center_thresh)
        thresh_valid = [] # select only thresholds with data in t range

        eff_all = []
        err_all = []
        rates = []
        rates_err = []

        for t in thresh_unique:
            t_millis = center_millis[center_thresh == t]
            
            # exclude data points outside t range
            if t_millis.min() > args.tmax or t_millis.max() < args.tmin:
                thresh_valid.append(False)
                continue

            thresh_valid.append(True)
            tmin = max(args.tmin, t_millis.min())
            tmax = min(args.tmax, t_millis.max())

            eff, err = get_eff(a, b, c, center, t_range=(tmin, tmax), verbose=args.thresh_verbose)
            eff_all.append(eff)
            err_all.append(err)

            if center == 'a':
                counts = np.sum((a > tmin) & (a < tmax))
            elif center == 'b':
                counts = np.sum((b > tmin) & (b < tmax))
            elif center == 'c':
                counts = np.sum((c > tmin) & (c < tmax))

            duration = (tmax - tmin) / 1000
            rates.append((counts - 1) / duration)
            rates_err.append(np.sqrt(counts - 1) / duration)

        thresh = thresh_unique[thresh_valid]

        ax = plt.gca()
        ax.errorbar(thresh, eff_all, yerr=err_all, c='b', marker='o', ms=3)
        
        ax2 = ax.twinx()
        ax2.errorbar(thresh, rates, yerr=rates_err, c='r', marker='o', ms=3)

        ax.set_xlabel('Threshold: {}'.format(center))
        ax.set_ylabel(r'$\epsilon$', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.set_ylabel('Count rate (Hz)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(bottom=0)
        plt.show()

    else:
        get_eff(a, b, c, 
            center, 
            t_range=(args.tmin, args.tmax), 
            verbose=True)

    npz.close()
