#!/usr/bin/env python

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

CONV_Ci2kBq = 3.7e7
COUNTS = {
        'max': 1,
        'sum5': 5,
        'sum9': 9,
        'sum21': 21,
        }


def get_photon_rate(bins, src, bg, src_tot, bg_tot, stat='max'): 

    src_n = np.array([src.GetEntries('{} >= {}'.format(stat, b)) for b in bins])
    bg_n = np.array([bg.GetEntries('{} >= {}'.format(stat, b)) for b in bins])

    p_src = src_n / src_tot
    p_bg = bg_n / bg_tot

    p_src_var = p_src * (1-p_src) / src_tot
    p_bg_var = p_bg * (1-p_bg) / bg_tot

    lambda_src = -np.log(1-p_src)
    lambda_bg = -np.log(1-p_bg)
    lambda_src_var = p_src_var / (1-p_src)**2
    lambda_bg_var = p_bg_var / (1-p_bg)**2

    lambda_photon = lambda_src - lambda_bg
    lambda_photon_var = lambda_src_var + lambda_bg_var

    return lambda_photon, np.sqrt(lambda_photon_var)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--src', required=True, help='ROOT file to be processed')
    parser.add_argument('--bg', required=True, help='ROOT file without source exposure') 
    parser.add_argument('--L', required=True, type=float, help='Distance between source and sensor in mm')
    parser.add_argument('--R', required=True, type=float, help='Source activity in Ci')
    parser.add_argument('--frame_duration', required=True, type=int, help='Frame duration in ms')
    parser.add_argument('--dL', type=float, help='Uncertainty in L')
    parser.add_argument('--dR', type=float, help='Uncertainty in R')

    parser.add_argument('--calib', required=True, help='Calibration directory')
    parser.add_argument('--bin_sz', type=int, default=1, help='Calibrated counts per bin')
    parser.add_argument('--thresh', type=int, default=1, help='Minimum ADC count to include')
    parser.add_argument('--sat', type=int, default=1023)
    parser.add_argument('--stat', default='max')
    parser.add_argument('--out', help='.npz to write graph data')
    parser.add_argument('--pix_pitch', type=float, help='Pixel pitch in microns')
    parser.add_argument('--xborder', type=int, default=0, help='Border used in find_muons.py')
    args = parser.parse_args()
 
    # get range from weights
    gmin = 0
    blk_lvl = 0
    try:
        gmin, blk_lvl, _ = load_electrons(args.calib)
        blk_lvl = int(blk_lvl)
    except FileNotFoundError:
        print('Could not find gain data.  Run pixelstats/electrons.py first')

    try:
        # find first-order correction for varying saturation level
        # from weighting, i.e. fraction of the sensor with saturation
        # below each calibrated value
        wgt = load_weights(args.calib)
        print(wgt.shape)
        npix = (wgt.shape[1]-2*args.xborder)*wgt.shape[0] if args.xborder else wgt.size
        hist, _ = np.histogram(wgt * (1023-blk_lvl) + 1, bins=np.arange(1025-blk_lvl))
        cumsum = np.cumsum(hist)[::args.bin_sz]
        sat_frac = cumsum[:-1] / cumsum[-1]
    except FileNotFoundError:
        print('Weights not found.  Using equal weights.')
        sat_frac = np.zeros((1023-blk_lvl) // args.bin_sz)
        width, height = load_res(args.calib)
        npix = (width - 2*args.xborder) * height

    bins = np.arange(args.bin_sz, 1024-blk_lvl, args.bin_sz) 

    f_src = r.TFile(args.src) 
    src = f_src.Get('triggers')
    src_tot = src.GetEntries() + f_src.Get('nontriggers').GetEntries()
    src_min = min([getattr(trig, args.stat) for trig in src]) - blk_lvl*COUNTS[args.stat]

    f_bg = r.TFile(args.bg)
    bg = f_bg.Get('triggers')
    bg_tot = bg.GetEntries() + f_bg.Get('nontriggers').GetEntries()
    bg_min = min([getattr(trig, args.stat) for trig in bg]) - blk_lvl*COUNTS[args.stat]

    n_gamma, n_gamma_err = get_photon_rate(bins+blk_lvl*COUNTS[args.stat], 
            src, bg, src_tot, bg_tot, stat=args.stat)

    f_src.Close()
    f_bg.Close()

    scale = CONV_Ci2kBq / (4*np.pi) * args.frame_duration
    a_eps = n_gamma / scale / args.R * args.L**2 / (1-sat_frac) / 1e6
    a_eps_err = a_eps * np.sqrt((n_gamma_err/n_gamma)**2 \
            + (args.dR/args.R)**2 + 4*(args.dL/args.L)**2) 

    sensor_area = args.pix_pitch**2 / 1e12 * npix
    print(sensor_area)

    eff = a_eps / sensor_area
    err = a_eps_err / sensor_area

    vmin = max(src_min, bg_min, args.thresh)
    bin_min = (vmin + args.bin_sz - 1) // args.bin_sz
    bin_max = (args.sat - blk_lvl) // args.bin_sz

    print(u'A\u03F5 = {0:.4f} \u00B1 {1:.4f} mm^2'.format(a_eps[bin_min]*1e6, a_eps_err[bin_min]*1e6))
    print(u'\u03F5 = {0:.4f} \u00B1 {1:.4f}'.format(eff[bin_min], err[bin_min]))
    print()

    fig = plt.figure(figsize=(6, 4.5))
    ax = plt.gca()

    ax.plot(bins[bin_min:bin_max] / gmin, a_eps[bin_min:bin_max], '-', linewidth=1)
    ax.fill_between(bins[bin_min:bin_max] / gmin, (a_eps-a_eps_err)[bin_min:bin_max], (a_eps+a_eps_err)[bin_min:bin_max], alpha=0.2, edgecolor=None)

    if args.out:
        np.savez(args.out,
               stat=args.stat,
               thresh=bins[bin_min:bin_max],
               electrons=bins[bin_min:bin_max]/gmin,
               eff=eff[bin_min:bin_max],
               err=err[bin_min:bin_max])

    ax.set_xlabel('Electrons')
    ax.set_ylabel(r'$A\epsilon$ $\mathrm{(m^2)}$') 

    plt.semilogy()

    plt.tight_layout()
    plt.show()
