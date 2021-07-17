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

def process_file(hodofile, pair=None):
    
    if pair:
        return process_pair(hodofile, pair)
    
    f = np.load(hodofile)

    thresh = []
    rate   = []
    err    = []
   
    tc = f.f.millis_c
    thresh_c = f.f.thresh_c
    interval_thresh = f.f.interval_c
    
    thr_vals = np.unique(thresh_c)
    thr_min, thr_max = args.thresh_range
    thr_vals = thr_vals[(thr_vals >= thr_min) & (thr_vals < thr_max)]

    for t in thr_vals:
        t_cut = tc[thresh_c == t]

        # get unbiased intervals to measure rates
        ti = f.f.interval_ti[interval_thresh == t]
        tf = f.f.interval_tf[interval_thresh == t]
        intervals = np.vstack([ti, tf]).T

        r, e = get_rate(t_cut, intervals)
        
        thresh.append(t)
        rate.append(r*1e6)
        err.append(e*1e6)

    return thresh, rate, err

    
def process_pair(hodofile, pair):
    pair = pair.lower()
    if not len(pair) == 2 or not set(pair).issubset(set('abc')): 
        raise ValueError('Invalid "pair" argument')

    f = np.load(hodofile)

    thresh = []
    rate   = []
    err    = []
    #prob   = []
    #perr   = []

    if 'a' in pair:
        t1 = f.f.millis_a
        thresh1 = f.f.thresh_a
        interval_thresh = f.f.interval_a

        if 'b' in pair:
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
        #p = t12_cut.size / t1_cut.size
        #prob.append(p)
        #perr.append((p*(1-p)/t1_cut.size)**0.5)

        #print(t, t1_cut.size, t2_cut.size, t12_cut.size, (rc-noise)*1e6)

    return thresh, rate, err


def add_voltage(ax):
    ax2 = ax.twiny()
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 40))
    ax2.set_xlabel('Threshold voltage [mV]')

    # set range
    xmin, xmax = ax.get_xlim()
    ax2.set_xlim(digit2dc(xmin), digit2dc(xmax))

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--hodofiles', nargs='+')
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--channels', nargs='+')

    parser.add_argument('--thresh_range', type=int, nargs=2, default=(0,256), help='Min and max thresholds')

    args = parser.parse_args()

    plt.figure(figsize=(5,4), tight_layout=True)
    plt.rc('font', size=12)
    ax = plt.gca()
    
    # add a single scintillator for reference
    #thresh_c, rate_c, err_c = process_file(args.hodofiles[0], None)
    #ax.errorbar(thresh_c, rate_c, yerr=err_c, marker='', label='Single PMT')

    second = False
    for f, p, label in zip(args.hodofiles, args.channels, args.labels):
        thresh, rate, err = process_file(f, p)

        if len(args.hodofiles) == 2 and second:
            ax2 = ax.twinx()
            ax2.errorbar(thresh, rate, yerr=err,
                    ls=':', marker='', color='indigo')
            ax2.tick_params(axis='y', labelcolor='indigo')
            ax2.set_ylabel(label, color='indigo')
            ax2.set_ylim(bottom=0)
            #ax2.semilogy()
    
        elif len(args.hodofiles) == 2:
            second = True
            ax.errorbar(thresh, rate, yerr=err,
                    ls=':', marker='', color='darkgreen')
            ax.tick_params(axis='y', labelcolor='darkgreen')
            ax.set_ylabel(label, color='darkgreen')
            ax.set_ylim(bottom=0)

        else:
            ax.errorbar(thresh, rate, yerr=err, color='k', ls=':', marker='', label=label)
            ax.set_ylabel('Coinc rate [mHz]')
            ax.legend()
            ax.semilogy()

    ax.set_xlabel('PWM threshold')
    
    add_voltage(ax)

    #plt.savefig('/home/jswaney/FishStandResults/systematics_rates.pdf', bbox_inches='tight')
    plt.show()
