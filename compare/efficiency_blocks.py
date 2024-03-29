#!/usr/bin/env python3

import os
import sys

import ROOT as r
import numpy as np
import matplotlib.pyplot as plt

# temporary hack to add pixelstats modules to path
fishfry_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(fishfry_dir, 'pixelstats'))
from geometry import load_res
from lens_shading import load_weights
from electrons import load_electrons

PAIRS = ['AB', 'AC', 'BC']
COUNTS = {
        'max': 1,
        'sum5': 5,
        'sum9': 9,
        'sum21': 21,
        }
COLZ = {
        'AB': 'r',
        'AC': 'g',
        'BC': 'b'
        }

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='')
    parser.add_argument('pfile', help='ROOT file generated by mark_triggered.py')
    parser.add_argument('--calib', default='calib', help='path to calibration directory')
    parser.add_argument('--err_n_rel', type=float, default=0, help='Relative error in acceptance due to variation in n')
    parser.add_argument('--thresh', type=int, default=0, help='Additional threshold')
    parser.add_argument('--bin_sz', type=int, default=1, help='Bin width for plots')
    parser.add_argument('--stat', choices=list(COUNTS.keys()), default='max', help='Statistic to measure')
    parser.add_argument('--sat', type=int, default=1023)
    parser.add_argument('--title', help='Plot title')
    parser.add_argument('--small', action='store_true', help='Display smaller plot')
    parser.add_argument('--log', action='store_true', help='Display a logscale for y')
    parser.add_argument('--xlim', type=float, nargs=2, help='Electron range to plot')
    parser.add_argument('--ylim', type=float, nargs=2, help='Efficiency range for plot')
    parser.add_argument('--out', help='Output .npz file for efficiency results')
    args = parser.parse_args() 


    # get range from weights
    gmin = 0
    blk_lvl = 0
    try:
        gmin, blk_lvl, dark_noise = load_electrons(args.calib)
        blk_lvl = int(blk_lvl)
    except FileNotFoundError:
        print('Could not find gain data.  Run pixelstats/electrons.py first')

    # now try loading calibration data
    f_acc = None
    try:
        f_acc = np.load(os.path.join(args.calib, 'acceptance.npz'))
        
    except:
        print('Acceptance not found. Try running "acceptance.py" first.')
        print('Using unit acceptance for testing.')


    try:
        # find first-order correction for varying saturation level
        # from weighting, i.e. fraction of the sensor with saturation
        # below each calibrated value
        wgt = load_weights(args.calib)
        hist, _ = np.histogram(wgt * (1023-blk_lvl) + 1, bins=np.arange(1025-blk_lvl))
        cumsum = np.cumsum(hist)[::args.bin_sz]
        sat_frac = cumsum[:-1] / cumsum[-1]

    except FileNotFoundError:
        print('Weights not found.  Using equal weights.')
        sat_frac = np.zeros((1024-blk_lvl) // args.bin_sz - 1)
 
    # finally, look through ROOT files

    pf = r.TFile(args.pfile)

    # in order to avoid artificial caps, we restrict the range
    # to the saturation value even on sum statistics 
    
    figsize = (4.5, 3.5) if args.small else (6, 4.5)
    fig = plt.figure(figsize=figsize)

    #ax1, ax2 = fig.suplots(1,2)
    ax1 = plt.gca()

    if args.log:
        ax1.semilogy()

    ax1.set_xlabel('Calibrated ADC counts')
    ax1.set_ylabel('Efficiency')    
    if args.title:
        ax1.set_title(args.title)

    if args.ylim:
        ax1.set_ylim(*args.ylim)


    #ax2.set_xlabel('Calibrated ADC counts')
    #ax2.set_ylabel('signal / noise')
    #ax2.set_title('Integrated signal-to-noise ratio by max values')

    npz_out = {}
    bins = np.arange(args.bin_sz, 1024-blk_lvl, args.bin_sz)
    for c in PAIRS:

        print(c)

        blocks = pf.Get('blocks_{}'.format(c))

        uinfo = blocks.GetUserInfo()
        n_hodo = t_tot = None

        for param in uinfo:
            if param.GetName() == 'n_hodo':
                n_hodo = param.GetVal()
            if param.GetName() == 't_tot':
                t_tot = param.GetVal()

        if not n_hodo or not t_tot:
            print("ERROR: Metadata not found")
            exit(1)

        # get acceptance data

        if f_acc:
            # P(hodoscope | phone)
            p_hgp = f_acc['p_hgp_{}'.format(c)]
            p_hgp_err = f_acc['p_hgp_err_{}'.format(c)]
            
            # P(phone | hodoscope)
            p_pgh = f_acc['p_pgh_{}'.format(c)]
            p_pgh_err = f_acc['p_pgh_err_{}'.format(c)]

            if args.err_n_rel:
                p_hgp_err = np.sqrt(p_hgp_err**2 + (args.err_n_rel*p_hgp)**2)
                p_pgh_err = np.sqrt(p_pgh_err**2 + (args.err_n_rel*p_pgh)**2)

            # in mm
            acc = f_acc['hodo_acc_{}'.format(c)]  

            # print hodoscope rates
            hodo_rate = n_hodo / t_tot * 1e3 # in Hz
            hodo_rate_err = np.sqrt(n_hodo) / t_tot * 1e3

            print(u'observed rate: {:.3f} \u00B1 {:.3f} mHz'.format(hodo_rate * 1e3, hodo_rate_err * 1e3))
            print('observed flux: {:.3f} Hz / m^2'.format(hodo_rate / acc * 1e6))
        else:
            print('No acceptance found')
            p_pgh = 1
            p_pgh_err = 0


        # histogram blocks
        vmin = args.sat # this could probably just be saved from thresholds  

        hist_MN = {}

        for iblk, blk in enumerate(blocks):
            print(iblk, '/', blocks.GetEntries(), end='\r')
            cmax = getattr(blk, args.stat) - blk_lvl*COUNTS[args.stat]
            cmax = np.clip(cmax, 0, args.sat-blk_lvl)
            if cmax > 0:
                vmin = min(vmin, cmax)
            
            idx_MN = (blk.M, blk.N)
            if not idx_MN in hist_MN:
                hist_MN[idx_MN] = np.zeros(bins.size+1, dtype=int)
            
            hist_MN[idx_MN][cmax // args.bin_sz] += 1
        

        # now extract efficiency
        cum_MN = {MN: np.cumsum(h[::-1])[::-1][1:] for MN, h in hist_MN.items()}
        n_MN = {MN: hist.sum() for MN, hist in hist_MN.items()}
        p_MN = {MN: cum_MN[MN] / n_MN[MN] for MN in cum_MN}
        p_noise = p_MN[(0,1)]
        n_untagged = hist_MN[(0,1)].sum()

        one_m_eff = 0
        n_tot = 0
        cache = []
        for MN, cum in cum_MN.items():
            print('({},{}): {} / {}\t\t'.format(*MN, cum[args.thresh//args.bin_sz], n_MN[MN]))
            
            M,N = MN
            if M != 1: continue
        
            one_m_eff_MN = (1 - p_MN[MN]) / (1 - p_noise)**N

            # weighted average
            one_m_eff += n_MN[MN] * one_m_eff_MN
            n_tot += n_MN[MN]

            # save for second pass
            cache.append((N, n_MN[MN]))

        eff_tot = 1 - one_m_eff / n_tot

        var1 = 0
        var2 = 0

        for N, n in cache:
            # calculate p_MN from the full dataset to avoid instabilities
            p = 1 - (1 - eff_tot) * (1 - p_noise)**N

            var1 += p * n / (1-p)
            var2 += n * N

        var_tot = ((1 - eff_tot) / n_tot)**2 * (var1 + p_noise / (1-p_noise) / n_untagged * var2**2)
        
        vmin = max(vmin, args.thresh)
        bin_min = (vmin + args.bin_sz-1) // args.bin_sz - 1
        bin_max = (args.sat-blk_lvl) // args.bin_sz

        eff = eff_tot / p_pgh / (1 - sat_frac)
        #print(bins[bin_min:200] / gmin)
        random_err = np.sqrt(var_tot) / p_pgh / (1 - sat_frac)
        print(np.interp([20,50,100], bins/gmin, random_err))
        tot_err = eff*np.sqrt(var_tot / eff_tot**2 + p_pgh_err**2 / p_pgh**2) 
        print(np.interp([20,50,100], bins/gmin, tot_err))

        #snr = (1/frac-1)**-1
        #snr_err = np.sqrt(frac_var)/(1-frac)**2 

        print(u'\u03F5A = {0:.4f} \u00B1 {1:.4f}'.format(eff_tot[bin_min], np.sqrt(var_tot)[bin_min]))
        print(u'A  = {0:.4f} \u00B1 {1:.4f}'.format(p_pgh, p_pgh_err))
        print(u'\u03F5  = {0:.4f} \u00B1 {1:.4f}'.format(eff[bin_min], tot_err[bin_min]))
        print()

        npz_out['eff_{}'.format(c)] = eff[bin_min:bin_max]
        npz_out['err_{}'.format(c)] = tot_err[bin_min:bin_max]

        ax1.plot(bins[bin_min:bin_max], eff[bin_min:bin_max], '-',
                linewidth=1, label=c, color=COLZ[c])
        #ax1.fill_between(bins[bin_min:], 
        #        (eff-random_err)[bin_min:], 
        #        (eff+random_err)[bin_min:], 
        #        alpha=0.15, color=COLZ[c], edgecolor=None)
        ax1.fill_between(bins[bin_min:bin_max], 
                (eff-tot_err)[bin_min:bin_max], 
                (eff+tot_err)[bin_min:bin_max], 
                alpha=0.2, color=COLZ[c], edgecolor=None)

        #ax2.plot(bins[bin_min:], snr[bin_min:], '-', label=c)
        #ax2.fill_between(bins[bin_min:], (snr-snr_err)[bin_min:], (snr+snr_err)[bin_min:], alpha=0.2)


    if f_acc:
        f_acc.close()
    pf.Close()
        
    if gmin:
        ax1twin = ax1.twiny()
        if args.xlim:
            ax1twin.set_xlim(*args.xlim) 
            ax1.set_xticks(np.arange(0,1024, 100))
            ax1.set_xlim(*(np.array(args.xlim) * gmin))
        else:
            ax1twin.set_xlim(*(np.array(ax1.get_xlim()) / gmin))
        
        ax1twin.set_xlabel('Electrons')
        ax1twin.xaxis.set_ticks_position('bottom')
        ax1twin.xaxis.set_label_position('bottom')
        ax1twin.spines['bottom'].set_position(('outward', 36))

        if args.out:
            for c in PAIRS:
                np.savez(args.out.replace('.npz', '_{}.npz'.format(c)), 
                        stat=args.stat, 
                        thresh=bins[bin_min:bin_max],
                        electrons=bins[bin_min:bin_max] / gmin,
                        eff=npz_out['eff_{}'.format(c)],
                        err=npz_out['err_{}'.format(c)])

        #ax2twin = ax2.twiny()
        #ax2twin.set_xlim(*(np.array(ax2.get_xlim()) / gmin))
        #ax2twin.set_xlabel('Electrons')
        #ax2twin.xaxis.set_ticks_position('bottom')
        #ax2twin.xaxis.set_label_position('bottom')
        #ax2twin.spines['bottom'].set_position(('outward', 36))


    ax1.legend()
    plt.tight_layout() 

    plt.show() 
