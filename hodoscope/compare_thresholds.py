#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from efficiency import load, get_eff_curve
from plot_coincidences import add_voltage

def fit_chi2(x1, y1, y1err, x2, y2, y2err, bins=None):
    xmin = x2.min()
    xmax = x2.max()

    if bins is None:
        bins = np.geomspace(0.67, 1.5, 100)
    
    chi2vals = []
    for scale in bins:
        # adjust x1 values
        x1_s = scale * x1
        xcut = (x1_s > xmin) & (x1_s < xmax)
        x1_s = x1_s[xcut]
        y1_s = y1[xcut]
        y1err_s = y1err[xcut]
        
        # find y2 values at new x1
        y2_adj = np.interp(x1_s, x2, y2)
        y2err_adj = np.interp(x1_s, x2, y2err) 

        chi2 = np.sum((y2_adj - y1_s)**2 / (y2err_adj**2 + y1err_s**2))
        chi2vals.append(chi2/xcut.sum())

    #plt.figure()
    #plt.plot(bins, chi2vals)

    return chi2vals, bins

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('infiles', nargs='+')
    parser.add_argument('center', choices=('a','b','c'))
    parser.add_argument('--target', nargs='+', required=True)

    args = parser.parse_args()
    
    a, b, c, tha, thb, thc = load(args.infiles)

    thresh, eff, eff_err, _, _ = get_eff_curve(
            (a, b, c),
            (tha, thb, thc),
            args.center)

    plt.figure(figsize=(5,4))
    ax = plt.gca()

    bins = np.geomspace(0.8, 1.4, 100)
    chi2tot = np.zeros(100)

    targetfiles = []
    centers = []

    target_c = []
    for s in args.target:
        if s in ('a','b','c'):
            centers.append(s)
            targetfiles.append(target_c)
            target_c = []

        else: target_c.append(s)

    for target, center in zip(targetfiles, centers):
        tgt_a, tgt_b, tgt_c, tgt_tha, tgt_thb, tgt_thc = load(target)
        thresh_t, eff_t, eff_err_t, _, _ = get_eff_curve(
            (tgt_a, tgt_b, tgt_c),
            (tgt_tha, tgt_thb, tgt_thc),
            center)

        chi2, bins = fit_chi2(thresh-52, eff, eff_err, 
                thresh_t-52, eff_t, eff_err_t,
                bins=bins)

        chi2tot += chi2 
        ax.errorbar(thresh_t, eff_t, yerr=eff_err_t, label=center.upper())

    scale = bins[np.argmin(chi2tot)]
    ax.errorbar((thresh-52)*scale + 52, eff, yerr=eff_err, label=args.center.upper() +" corrected")
    ax.set_xlabel('PWM threshold')
    ax.set_ylabel(r'$\epsilon$')
    add_voltage(ax)

    ax.legend()

    print(scale)

    plt.tight_layout()
    plt.show()
 
