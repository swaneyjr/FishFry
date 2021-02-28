#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def get_rate(times, intervals):
    count = 0
    t_tot = 0

    for ti, tf in intervals:
        count += np.sum((times > ti) & (times < tf))
        t_tot += tf - ti

    return count / t_tot, np.sqrt(count) / t_tot

def coincidences(t1, t2):
    dt = (-1, 0, 1)
    t_coinc = [np.intersect1d(t1, t2+t) for t in dt]
    return np.unique(np.hstack(t_coinc))

digit2dc = lambda digit: (digit-52)/255*5000

def add_voltage(ax):
    ax2 = ax.twiny()
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 40))
    ax2.set_xlabel('Threshold voltage (mV)')

    # set range
    xmin, xmax = ax.get_xlim()
    ax2.set_xlim(digit2dc(xmin), digit2dc(xmax))

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('hodofile')

    parser.add_argument('-a', action='store_true')
    parser.add_argument('-b', action='store_true')
    parser.add_argument('-c', action='store_true')

    parser.add_argument('--thresh_range', type=int, nargs=2, default=(0,256), help='Min and max thresholds')

    args = parser.parse_args()

    f = np.load(args.hodofile)

    thresh = []
    rate   = []
    err    = []
    prob   = []
    perr   = []
    #rate1  = []
    #err1   = []
    #rate2  = []
    #err2   = []

    if args.a:
        t1 = f.f.millis_a
        thresh1 = f.f.thresh_a
        interval_thresh = f.f.interval_a

        if args.b:
            t2 = f.f.millis_b
            thresh2 = f.f.thresh_b
            if not np.all(f.f.interval_a == f.f.interval_b):
                raise ValueError('Only runs with equal thresholds are supported')

        else:
            t2 = f.f.millis_c
            thresh2 = f.f.thresh_c
            if not np.all(f.f.interval_a == f.f.interval_c):
                raise ValueError('Only runs with equal thresholds are supported')
    else:
        t1 = f.f.millis_b
        thresh1 = f.f.thresh_b
        t2 = f.f.millis_c
        thresh2 = f.f.thresh_c
        interval_thresh = f.f.interval_b
        if not np.all(f.f.interval_b == f.f.interval_c):
            raise ValueError('Only runs with equal thresholds are supported')

    
    thr_vals = np.unique(thresh1)
    thr_min, thr_max = args.thresh_range
    thr_vals = thr_vals[(thr_vals >= thr_min) & (thr_vals < thr_max)]

    for t in thr_vals:
        t1_cut = t1[thresh1 == t]
        t2_cut = t2[thresh2 == t]
        t12_cut = coincidences(t1_cut, t2_cut)

        # get unbiased intervals to measure rates
        ti = f.f.interval_ti[interval_thresh == t]
        tf = f.f.interval_tf[interval_thresh == t]
        intervals = np.vstack([ti, tf]).T

        r1, e1 = get_rate(t1_cut, intervals)
        r2, e2 = get_rate(t2_cut, intervals)
        rc, ec = get_rate(t12_cut, intervals)
        noise = r1*r2*3 # small enough that error contribution is negligible

        thresh.append(t)
        rate.append((rc - noise)*1e6)
        err.append(ec*1e6)
        #rate1.append(r1*1e6)
        #err1.append(e1*1e6)
        #rate2.append(r2*1e6)
        #err2.append(e2*1e6)
        p = t12_cut.size / t1_cut.size
        prob.append(p)
        perr.append((p*(1-p)/t1_cut.size)**0.5)

        print(t, t1_cut.size, t2_cut.size, t12_cut.size, (rc-noise)*1e6)


    plt.figure(figsize=(4,4))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.errorbar(thresh, rate, yerr=err, marker='', color='green')
    ax2.errorbar(thresh, prob, yerr=perr, marker='', color='purple')
    #ax2.errorbar(thresh, rate1, yerr=err1, marker='o', color='r')
    #ax2.errorbar(thresh, rate2, yerr=err2, marker='o', color='r')

    ax1.set_ylim(bottom=0)    
    ax2.set_ylim(bottom=0)

    ax1.tick_params(axis='y', labelcolor='green')
    ax2.tick_params(axis='y', labelcolor='purple')

    ax1.set_xlabel('PWM threshold')
    ax1.set_ylabel('Coinc rate (mHz)', color='green')
    #ax2.set_ylabel('Total rate (mHz)', color='r')
    ax2.set_ylabel('Coinc probability', color='purple')

    add_voltage(ax1)

   
    plt.tight_layout()
    plt.show()
