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
    blk  = f_electrons['blk_lvl']
    dark = f_electrons['dark_noise']

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


def plot_2x2(palettes, f, *args, superimpose=False, **kwargs):
 
    plt.figure(figsize=(10,8))

    for ij in range(4): 

        i = ij %  2
        j = ij // 2

        cargs = [a[j, i] for a in args]

        if superimpose:
            f(palettes[ij], *cargs, **kwargs)
        else:
            plt.subplot(2,2,ij+1)
            f(palettes[ij], *cargs, **kwargs)


def plot_meanvar(palette, cmean, cvar, g, off):
    # plot cvar vs. cmean and gain vs raw mean
    plt.hist2d(cmean.flatten(), cvar.flatten(), bins=(500,500), range=((0,1024),(0,2000)), norm=LogNorm(), cmap=palette.cmap)
    plt.xlabel('Adjusted mean')
    plt.ylabel('Adjusted variance')
    plt.colorbar()
    plt.plot(np.arange(800), np.arange(800)*g + off, 'y-')
    

def plot_gain_mu(palette, mu, g_pt, g):
    plt.hist2d(mu.flatten(), g_pt.flatten(), bins=(500, 500), range=((0,1024), (0,3)), norm=LogNorm(), cmap=palette.cmap)
    plt.xlabel('Sample mean')
    plt.ylabel('Gain')
    plt.colorbar()
    plt.plot(np.arange(800), np.repeat(g, 800), 'y-')
    

def plot_gain_xy(palette, g, vmin, vmax):
    plt.imshow(np.nanmean(g,axis=0), cmap=palette.cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()

def plot_gain_hist(palette, g, bins=None, vmin=0, vmax=7):
    plt.hist(g.flatten(), bins=bins, color=palette.color, histtype='step')
    plt.title('Gain values by channel')    
    plt.xlabel('Gain (DN/e-)')
    plt.ylabel('Pixel count')
    plt.loglog()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('infiles', nargs='+', help='Pixelstats calibration files')

    parser.add_argument('--calib', help='path to calibration directory')
    parser.add_argument('--color_filter', choices=('RGGB','GRBG','GBRG','BGGR'), help='color filter arrangement')
    parser.add_argument('--black', type=float, default=0, help='Estimate of black level offset')
    parser.add_argument('--commit', action='store_true', help='store result to .npz') 

    args = parser.parse_args()

    wgt = np.ones((2,2))

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
    wgt   = np.moveaxis(wgt.reshape(-1, height//2, 2, width//2, 2), (2,4),(0,1))


    fit_gain = []
    fit_offset = []
    for iy in range(2):
        fit_gain.append([])
        fit_offset.append([])
        for ix in range(2):
            cmean_ij = cmean[iy, ix].flatten()
            cvar_ij = cvar[iy, ix].flatten()

            cut = (cmean_ij > 40) & (cmean_ij < 500) & (cvar_ij < 3000)
            cmean_cut = cmean_ij[cut]
            cvar_cut = cvar_ij[cut]

            gain, offset = fit(cmean_cut, cvar_cut)
            
            fit_gain[iy].append(gain)
            fit_offset[iy].append(offset)

    fit_gain = np.array(fit_gain)
    fit_offset = np.array(fit_offset)
    
    print(fit_gain)
    print(fit_offset)

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

    gain_all = np.mean(fit_gain)
    dark_noise = np.mean(fit_offset - 1/12) / gain_all
    
    print('Gain:')
    print(fit_gain)
    print()
    print('Dark noise:')
    print(dark_noise)

    g_pt = (cvar - dark_noise*(fit_gain.reshape(2,2,1,1,1)/wgt + 1/12)**2) / (cmean-blk_diff)

    # get range for bins?
    
    gsorted = np.sort(g_pt.flatten())
    vmin = gsorted[gsorted.size//100]
    vmax = gsorted[-gsorted.size//100]
    bins = np.geomspace(vmin, vmax, 100)

    # make plots

    palettes = Palette.make(args.color_filter)
    plot_2x2(palettes, plot_meanvar, cmean, cvar, fit_gain, fit_offset)
    plot_2x2(palettes, plot_gain_mu, mean, g_pt, fit_gain)
    plot_2x2(palettes, plot_gain_xy, g_pt, vmin=vmin, vmax=vmax)
    plot_2x2(palettes, plot_gain_hist, g_pt, superimpose=True, bins=bins, vmin=vmin, vmax=vmax)

    plt.show()

    if args.commit: 

        np.savez(os.path.join(args.calib, 'electrons.npz'),
                gain=fit_gain,
                blk_lvl=blk_lvl,
                dark_noise=dark_noise)
