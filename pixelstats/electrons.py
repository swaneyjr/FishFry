#!/usr/bin/env python

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from geometry import load_res
from lens_shading import load_weights

def load_electrons(calib):
    f_electrons = np.load(os.path.join(calib, 'electrons.npz'))

    gain = np.mean(f_electrons['gain'])
    dark = np.mean(f_electrons['dark_noise'])
    blk  = f_electrons['blk_lvl']

    return gain, blk, dark


# utility class for plotting
class Palette:
    cmaps = {
        'r': 'Reds',
        'g': 'Greens',
        'b': 'Blues',
        'k': 'Greys'
        }

    def __init__(self, c):
        self.color = c.lower()
        self.cmap = Palette.cmaps[self.color]

    @staticmethod
    def make(cfilter):
        if cfilter:
            return [Palette(c) for c in cfilter]
        else:
            return [Palette('K') for _ in range(4)]


def correct_stats(mean, var, wgt, blk_lvl):
    quant_noise = 1/12

    cmean = (mean-blk_lvl) * wgt
    cvar = (var-quant_noise) * wgt**2 + quant_noise
     
    return cmean, cvar


def fit(cmean, cvar, nbins=100):

    g0 = np.nanmean(cvar/cmean)

    cmean = cmean[::7]
    cvar = cvar[::7]

    # pseudo-Hough transform
    slope_space  = np.linspace(g0-0.2, g0+0.2, nbins)
    offset_space = np.linspace(-10, 10, nbins)

    slope_bins = np.linspace(g0-0.2, g0+0.2, nbins+1)
    offset_bins = np.linspace(-10, 10, nbins+1)

    offsets = (np.repeat(offset_space.reshape(-1,1), cmean.size, axis=1)).flatten()
    slopes = ((cvar - offset_space.reshape(-1,1)) / cmean).flatten()
    accumulator = np.histogram2d(offsets, slopes, bins=(offset_bins, slope_bins))[0]
    #plt.imshow(accumulator.T, extent=[offset_bins[0], offset_bins[-1], slope_bins[0], slope_bins[-1]], aspect='auto', origin='lower')
    #plt.show()

    max_bin = np.argmax(accumulator)
    peak_slope = slope_space[max_bin % nbins]
    peak_offset = offset_space[max_bin // nbins]

    return peak_slope, peak_offset


def plot_2x2(palettes, f, *args, superimpose=False, small=False, pick1=-1, **kwargs):
 
    figsize = (3.8,3) if small else (10,8)
    if pick1 >= 0 or superimpose:
        plt.figure(figsize=figsize, tight_layout=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize, tight_layout=True)

    for ij in range(4): 
        if pick1 >= 0 and pick1 != ij: continue 
        
        i = ij %  2
        j = ij // 2

        cargs = [a[j, i] for a in args]
        ax = plt.gca() if pick1 >= 0 or superimpose else axes[j, i]

        if superimpose or pick1 >= 0:
            f(ax, palettes[ij], *cargs, **kwargs)
        else:
            f(ax, palettes[ij], *cargs, **kwargs)

def plot_meanvar(ax, palette, cmean, cvar, g, off):
    # plot cvar vs. cmean and gain vs raw mean
    ax.hist2d(cmean.flatten(), cvar.flatten(), bins=(500,500), range=((0,1024),(0,1500)), norm=LogNorm(), cmap=palette.cmap)
    ax.set_xlabel(r'$\mu\,/\,\lambda(r)$')
    ax.set_ylabel(r'$\sigma^2\,/\,\lambda(r)^2$')
    #plt.colorbar()
    ax.plot(np.arange(800), np.arange(800)*g + off, 'y-')
    

def plot_gain_mu(ax, palette, mu, g_pt, g):
    ax.hist2d(mu.flatten(), g_pt.flatten(), bins=(500, 500), range=((0,1024), (0,3)), norm=LogNorm(), cmap=palette.cmap)
    ax.set_xlabel('Sample mean')
    ax.set_ylabel('Gain')
    #plt.colorbar()
    ax.plot(np.arange(800), np.repeat(g, 800), 'y-')
    

def plot_gain_xy(ax, palette, g, vmin, vmax):
    ax.imshow(np.nanmean(g,axis=0), cmap=palette.cmap, vmin=vmin, vmax=vmax)
    #plt.colorbar()

def plot_gain_hist(ax, palette, g, bins=None, vmin=0, vmax=7):
    ax.hist(g[np.logical_not(np.isnan(g))].flatten(), bins=bins, color=palette.color, histtype='step')
    ax.set_title('Gain values by channel')    
    ax.set_xlabel('Gain [DN/e-]')
    ax.set_ylabel('Pixel count')
    ax.loglog()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('infiles', nargs='+', help='Pixelstats calibration files')

    parser.add_argument('--calib', help='path to calibration directory')
    parser.add_argument('--color_filter', choices=('RGGB','GRBG','GBRG','BGGR'), help='color filter arrangement')
    parser.add_argument('--black', type=float, default=0, help='Estimate of black level offset')
    parser.add_argument('--small', action='store_true', help='Generate small plots')
    parser.add_argument('--pick1', type=int, default=-1, help='Plot only one channel')
    parser.add_argument('--commit', action='store_true', help='store result to .npz') 

    args = parser.parse_args()

    wgt = np.array([1])

    try:
        wgt = load_weights(args.calib)
        print('{} weights found in interval [{:.3f}, {:.3f}]'.format(wgt.size, wgt.min(), wgt.max()))
    except FileNotFoundError:
        print('Could not find weights... using equal weights') 

    width, height = load_res(args.calib)

    mean = []
    var  = []

    for fname in args.infiles: 
        f = np.load(fname)
        
        sum_ = f['sum']
        ssq  = f['ssq']
        num  = f['num'] 

        mean_f = sum_ / num
        var_f  = (ssq / num - mean_f**2) * num / (num-1)

        mean.append(mean_f.reshape(height, width))
        var.append(var_f.reshape(height,width))

    # combine and organize by color filter
    mean = np.array(mean)    
    var  = np.array(var)

    cmean, cvar = correct_stats(mean, var, wgt, args.black)
    
    # reshape so that the color axes are in front
    mean  = np.moveaxis(mean.reshape(-1, height//2, 2, width//2, 2), (2,4),(0,1))
    var   = np.moveaxis(var.reshape(-1, height//2, 2, width//2, 2), (2,4),(0,1))
    cmean = np.moveaxis(cmean.reshape(-1, height//2, 2, width//2, 2), (2,4),(0,1))
    cvar  = np.moveaxis(cvar.reshape(-1, height//2, 2, width//2, 2), (2,4),(0,1))


    fit_gain = []
    fit_offset = []
    for iy in range(2):
        fit_gain.append([])
        fit_offset.append([])
        for ix in range(2):
            cmean_ij = cmean[iy, ix].flatten()
            cvar_ij = cvar[iy, ix].flatten()

            cut = (cmean_ij > args.black + 40) & (cmean_ij < 500) & (cvar_ij < 3000)
            cmean_cut = cmean_ij[cut]
            cvar_cut = cvar_ij[cut]

            gain, offset = fit(cmean_cut, cvar_cut)
            
            fit_gain[iy].append(gain)
            fit_offset[iy].append(offset)

    fit_gain = np.array(fit_gain)
    fit_offset = np.array(fit_offset)

    # now intersect these lines to find black level and dark noise

    # var_tot = g**2 * var_dark + g * (mu - mu_dark) + 1/12

    #N = 4
    #bg  = np.sum(fit_offset * fit_gain)
    #bg2 = np.sum(fit_offset * fit_gain**2)
    #sg  = np.sum(fit_gain)
    #sg2 = np.sum(fit_gain**2)
    #sg3 = np.sum(fit_gain**3)
    #sg4 = np.sum(fit_gain**4)

    #blk_diff = ((bg2 - N*sg2/12)*sg3 - (bg - N*sg/12)*sg4) / (sg4*sg2 - sg3**2)
    #dark_noise = (bg + N*sg/12 + blk_diff*sg2) / sg3
    #print(blk_diff, dark_noise)

    blk_diff = 0
    blk_lvl = args.black + blk_diff

    dark_noise = (fit_offset - 1/12) / fit_gain**2
    
    print('Gain:')
    print(fit_gain)
    print()
    print('Dark noise:')
    print(dark_noise)

    g_pt = (cvar - dark_noise.reshape(2,2,1,1,1)*(fit_gain.reshape(2,2,1,1,1) + 1/12)**2) / (cmean-blk_diff)

    # get range for bins
    
    gsorted = np.sort(g_pt.flatten())
    gsorted = gsorted[gsorted>0]
    vmin = gsorted[gsorted.size//100]
    vmax = gsorted[-gsorted.size//100]
    bins = np.geomspace(vmin, vmax, 100)

    # make plots

    palettes = Palette.make(args.color_filter)
    plot_2x2(palettes, plot_meanvar, cmean, cvar, fit_gain, fit_offset, 
            small=args.small, pick1=args.pick1)
    plot_2x2(palettes, plot_gain_mu, mean, g_pt, fit_gain,
            small=args.small, pick1=args.pick1)
    
    # clean out saturated values for next plots
    g_pt[(cmean > 600) | (g_pt < 0)] = np.nan

    plot_2x2(palettes, plot_gain_xy, g_pt, 
            vmin=vmin, vmax=vmax, small=args.small, pick1=args.pick1)
    plot_2x2(palettes, plot_gain_hist, g_pt, superimpose=True, bins=bins, 
            vmin=vmin, vmax=vmax, small=args.small, pick1=args.pick1)

    plt.show()

    if args.commit: 

        np.savez(os.path.join(args.calib, 'electrons.npz'),
                gain=fit_gain,
                blk_lvl=blk_lvl,
                dark_noise=dark_noise)

