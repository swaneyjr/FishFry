#!/usr/bin/python

import os
import sys

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# temporary hack to add pixelstats modules to path
fishfry_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(fishfry_dir, 'monte_carlo'))

from plot_coincidences import coincidences, get_rate, add_voltage
from acceptance import monte_carlo  

def load(f, c):
    c1, c2 = sorted(c)
    if c1 == 'a':
        thresh1 = f['thresh_a']
        millis1 = f['millis_a']
        interval_thresh = f['interval_a']
        if c2 == 'b':
            thresh2 = f['thresh_b']
            millis2 = f['millis_b']
            if not np.all(f['interval_a'] == f['interval_b']):
                raise ValueError('Only runs with equal thresholds are supported')

        else:
            thresh2 = f['thresh_c']
            millis2 = f['millis_c']
            if not np.all(f['interval_a'] == f['interval_c']):
                raise ValueError('Only runs with equal thresholds are supported')


    else:
        thresh1 = f['thresh_b']
        thresh2 = f['thresh_c']
        millis1 = f['millis_b']
        millis2 = f['millis_c']
        interval_thresh = f['interval_b']
        if not np.all(f['interval_b'] == f['interval_c']):
            raise ValueError('Only runs with equal thresholds are supported')



    thresh_unique = np.unique(np.hstack([thresh1, thresh2]))
    rate = []
    err  = []
    for thr in thresh_unique:
        t1 = millis1[thresh1 == thr]
        t2 = millis2[thresh2 == thr]
       
        # get unbiased intervals to measure rates
        ti = f['interval_ti'][interval_thresh == thr]
        tf = f['interval_tf'][interval_thresh == thr]
        intervals = np.vstack([ti, tf]).T

        thr_rate, thr_err = get_rate(coincidences(t1, t2), intervals)

        rate.append(thr_rate)
        err.append(thr_err)

    return thresh_unique, rate, err


def mc_ratio(thresh, gap1, gap2, n=2):
    # LYSO dimensions
    x = np.array([14., 14.])
    y = np.array([16., 16.])
    
    z1 = np.array([0., gap1])
    z2 = np.array([0., gap2])

    border = 3 * np.tan(45*np.pi/180)

    r_vals = []

    for _ in range(100):

        dx1, dx2, dy1, dy2, dz1, dz2 = np.random.normal(0, 0.1, 6)

        hits1 = monte_carlo(200000, 
                -x/2+dx1, x/2+dx1, 
                -y/2+dy1, y/2+dy1,
                z1-3+dz1, z1+3+dz1,
                0, border, thresh, n_ang=n)
        
        hits2 = monte_carlo(200000, 
                -x/2+dx2, x/2+dx2, 
                -y/2+dy2, y/2+dy2,
                z2-3+dz2, z2+3+dz2,
                0, border, thresh, n_ang=n)

        n1 = np.logical_and.reduce(hits1).sum()
        n2 = np.logical_and.reduce(hits2).sum()

        r_vals.append(n2/n1)

    return np.mean(r_vals), np.std(r_vals)


def make_ratio_plots(exp_thresh, exp_ratio, exp_err, theory_thresh=None, theory_ratio=None, theory_err=None, plot_all=True, out=None):
    
    if plot_all:
        ax_r = plt.subplot(122)
    else:
        ax_r = plt.gca()
    ax_r.errorbar(exp_thresh, exp_ratio, yerr=exp_err, ls='', marker='o', color='k', label='Data')

    def ratio2thresh(r):
        return np.interp(r, np.hstack([[0], theory_ratio, [1]]), np.hstack([[-1], theory_thresh, [7]]))
    
    def thresh2ratio(t):
        return np.interp(t, np.hstack([[-1], theory_thresh, [7]]), np.hstack([[0], theory_ratio, [1]]))

    ax_r.set_xlabel('PMT threshold')
    ax_r.set_ylabel('Coincidence rate ratio')
    
    if theory_ratio is None:
        return ax_r

    ax2y = ax_r.secondary_yaxis('right', functions=(ratio2thresh, thresh2ratio))
    ax2y.set_yticks(theory_thresh[::2])
    ax2y.set_yticks(theory_thresh[1::2], minor=True)
    ax2y.set_ylabel('Path length thresh. [mm]') 

    # do weighted least squares fit
    exp_path = ratio2thresh(exp_ratio)
    w = (exp_path >= 0) * (exp_path <= 6) * exp_err**-2
    
    x = exp_thresh - 52 # this gets us the real intercept
    y = exp_path

    wx = np.sum(w*x)
    wy = np.sum(w*y)
    wx2 = np.sum(w*x**2)
    wy2 = np.sum(w*y**2)
    wxy = np.sum(w*x*y)

    denom = w.sum() * wx2 - wx**2
    a = (wx2*wy - wx*wxy) / denom
    b = (w.sum()*wxy - wx*wy) / denom

    cov = np.array([[wx2, -wx],[-wx, w.sum()]]) / denom
    print(cov)

    print('intercept = {:.2f} \u00B1 {:.2f}'.format(a, np.sqrt(cov[0,0])))
    print('slope     = {:.3f} \u00B1 {:.3f}'.format(b, np.sqrt(cov[1,1])))

    ax_r.plot(exp_thresh[w>0], thresh2ratio(a + b*x[w>0]), 'k--', label='Linear fit')

    ymin, ymax = ax_r.get_ylim()
    if ymin > min(theory_ratio):
        ax_r.set_ylim(bottom=min(theory_ratio))
    if ymax < max(theory_ratio):
        ax_r.set_ylim(top=max(theory_ratio))

    if np.any(w>0):
        plt.legend(loc='upper left')

    if not np.any(w>0):
        return ax_r, None, None

    ax_theory = plt.subplot(121)
    ax_theory.plot(theory_thresh, theory_ratio, 'k')
    ax_theory.fill_between(theory_thresh, 
            theory_ratio-theory_err, 
            theory_ratio+theory_err,
            color='k', alpha=0.4)
    ax_theory.set_xlabel('Path length thresh. [mm]')
    ax_theory.set_ylabel('Coincidence rate ratio')

    #ax_lin = plt.subplot(133)

    #lin_err_lo = exp_path - ratio2thresh(exp_ratio - exp_err)
    #lin_err_hi = ratio2thresh(exp_ratio + exp_err) - exp_path
    #ax_lin.errorbar(exp_thresh[w>0], exp_path[w>0], yerr=(lin_err_lo, lin_err_hi), ls='', marker='o', color='k', label='Data')
    #ax_lin.plot(exp_thresh[w>0], a + b*x[w>0], 'k--', label='Linear fit')
    #ax_lin.set_xlabel('PMT threshold')
    #ax_lin.set_ylabel('Path length thresh. [mm]')
    #ax_lin.set_ylim(0, 6)
    #plt.legend(loc='upper left')

    # do a chi^2 test for good measure
    
    chi2 = np.sum(w*(thresh2ratio(a + b*x[w>0]) - exp_ratio[w>0])**2)
    df = (w>0).sum() - 2

    print('\u1D6A2 = {:.3f}'.format(chi2))
    print('df =', df)
    print('p  = {:.3f}'.format(stats.chi2.cdf(chi2, df)))
    
    if out:
        np.savez(out, coeffs=[a, b], cov=cov)

    return ax_r, ax_theory, None #ax_lin


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Compute effective distance threshold range given fluxes')

    parser.add_argument('--files', nargs=2, required=True, help='npz files with  hodoscope data')
    parser.add_argument('--gaps', nargs=2, type=float, required=True, help='distances between LYSO centers in mm in each file')
    parser.add_argument('--channels', nargs=2, required=True, help='strings of channels used (example: --channels AB ac)')
    parser.add_argument('--no_theory', action='store_true', help='skip computing theoretical values')
    parser.add_argument('--n', type=float, default=2, help='use cos^n(theta) dependence for flux')
    parser.add_argument('--out', help='Output with path length to voltage linear fit')

    args = parser.parse_args()

    # load hodoscope data
    f1, f2 = map(np.load, args.files)
    c1, c2 = map(str.lower, args.channels)

    thresh1, rate1, err1 = load(f1, c1)
    thresh2, rate2, err2 = load(f2, c2)

    exp_thresh, idx1, idx2 = np.intersect1d(thresh1, thresh2, return_indices=True)
    rate1 = np.array(rate1)[idx1]
    rate2 = np.array(rate2)[idx2]

    err1  = np.array(err1)[idx1]
    err2  = np.array(err2)[idx2]

    exp_ratio = rate2 / rate1
    exp_err   = exp_ratio * np.sqrt((err1/rate1)**2 + (err2/rate2)**2) 
    
    #print(err1/rate1, err2/rate2, exp_err/exp_ratio)

    # for testing, save time on MC
    if args.no_theory:
        make_ratio_plots(exp_thresh, exp_ratio, exp_err)
        plt.show()
        quit()

    # now generate MC ratios for comparison
    theory_thresh = np.linspace(0,6,13)
    theory_ratio = []
    theory_err = []
    for thresh in theory_thresh:
        r_thr, r_err = mc_ratio(thresh, *args.gaps, args.n)
        print(thresh, r_thr)
        theory_ratio.append(r_thr)
        theory_err.append(r_err)
    theory_ratio = np.array(theory_ratio)
    theory_err = np.array(theory_err)

    # find corresponding path length thresholds and compute flux
    exp_path = np.interp(exp_ratio, theory_ratio, theory_thresh)
    cut = (exp_path > 0) & (exp_path < 6)

    # check if the theory and experimental values intersect
    plot_all = exp_ratio.min() < theory_ratio.max() and exp_ratio.max() > theory_ratio.min()

    figsize = (8,4) if plot_all else (4,4)
    plt.figure(1, figsize=figsize, tight_layout=True)
    plt.rc('font', size=13)
    ax_r, ax_theory, ax_lin = make_ratio_plots(
            exp_thresh, exp_ratio, exp_err, 
            theory_thresh, theory_ratio, theory_err,
            plot_all, args.out)
    add_voltage(ax_r)
    if ax_lin:
        add_voltage(ax_lin)

    #plt.subplots_adjust(wspace=.1)

    plt.show()

