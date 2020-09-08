#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def plot_thresh(timestamps, thresh, label=None):
    thresh_range = np.arange(thresh.min()+1, thresh.max())
    rates = []
    err = []

    for thr in thresh_range:
        thr_times = timestamps[thresh == thr]
        counts = thr_times.size
        duration = (thr_times.max() - thr_times.min()) / (1 - 2/counts)
       
        rates.append(counts / duration)
        err.append(counts**0.5 / duration)

    plt.errorbar(thresh_range, rates, yerr=err, 
            marker='o', ms=3, ls='', label=label)
    
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('infile', help='npz file with trigger times')
    parser.add_argument('--theory', type=float, nargs='+', help='Theoretical detector flux in counts/min')
    args = parser.parse_args()

    f = np.load(args.infile)

    plot_thresh(f.f.micros_a, f.f.thresh_a, label='A')
    plot_thresh(f.f.micros_b, f.f.thresh_b, label='B')
    plot_thresh(f.f.micros_c, f.f.thresh_c, label='C')

    if args.theory:
        all_thresh = np.hstack([f.f.thresh_a, f.f.thresh_b, f.f.thresh_c])
        theory_x = np.arange(all_thresh.min(), all_thresh.max() + 1)

        for phi_vert in args.theory:
            # use a monte carlo obtained conversion
            theory_rate = phi_vert / 71 * 2.095
            theory_y = np.ones(theory_x.size) * theory_rate
            label = r'{:.1f} $m^{{-2}} s^{{-1}} sr^{{-1}}$'.format(phi_vert)
            plt.plot(theory_x, theory_y, label=label)

    plt.title('Counts vs. PWM thresholds')
    plt.xlabel('Threshold')
    plt.ylabel('Rate (counts/min)')

    plt.legend()
    plt.ylim(bottom=0)
    plt.show()
