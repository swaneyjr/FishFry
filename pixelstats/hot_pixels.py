#!/usr/bin/env python3
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from argparse import ArgumentParser
import numpy as np
import os

from geometry import load_res
from dark_pixels import load_dark
from lens_shading import load_weights

def load_hot(calib_dir, offline=False):
    f_online = np.load(os.path.join(calib_dir, 'hot_online.npz'))
    hot = f_online['hot_list']
    f_online.close()

    if offline:
        try:
            f_offline = np.load(os.path.join(calib_dir, 'hot_offline.npz'))
            off = f_offline['hot_list']
            hot = np.unique(np.hstack([hot, off]))
            f_offline.close()
        except IOError:
            print('No offline hot pixels to load')

    return hot

def export(hot_cells, outfile):
    print('exporting hotcells as .npz file')
    flat_hotcell_list = hot_cells.flatten()
    np.savez(outfile, hot_list=flat_hotcell_list)
    print('done.')


def plot(mean, variance, snd_max, cut_mean, cut_variance, cut_snd_max, vmax):
    print('\nplotting figures:')
    
    figsize = (4,3) if args.small else (12,6)
    plt.figure(figsize=figsize, tight_layout=True)
    if not args.small:
        plt.subplot(121) 
        plt.title('Before cut')
    cut = np.random.randint(mean.size, size=100000)
    plt.scatter(mean[cut], variance[cut], c=snd_max[cut], s=0.5, cmap='rainbow', vmax=vmax)
    #plt.hist2d(mean, variance, bins=500, norm=LogNorm())
    plt.xlabel(r'$\mu\,/\,\lambda(r)$')
    plt.ylabel(r'$\sigma^2\,/\,\lambda(r)^2$')
    plt.colorbar()

    if args.small:
        plt.figure(figsize=figsize, tight_layout=True)
    else:
        plt.subplot(122)
        plt.title('After cut')
    cut = np.random.randint(cut_mean.size, size=100000)
    plt.scatter(cut_mean[cut], cut_variance[cut], c=cut_snd_max[cut], s=0.5, cmap='rainbow', vmax=vmax)
    #plt.hist2d(cut_mean, cut_variance, bins=500, norm=LogNorm()) 
    plt.xlabel(r'$\mu\,/\,\lambda(r)$')
    plt.ylabel(r'$\sigma^2\,/\,\lambda(r)^2$')
    plt.colorbar()
  
    plt.show()
    print('done.')


def compute(data):
    
    sum_ = data['sum']
    num_ = data['num']
    ssq_ = data['ssq']
    snd  = data['second']

    data.close()

    mean = sum_ / num_
    var = (( ssq_ / num_ ) - ( (sum_ / num_)**2 )) * (( num_ / (num_ - 1) ))
    total_res = mean.size
    indices = np.arange(total_res)

    if not args.raw:
        try:
            wgt = load_weights(args.calib).flatten()
            mean *= wgt
            var *= wgt**2
            snd *= wgt
        except IOError:
            print('lens.npz not found')

    try:
        cut = np.logical_not(load_dark(args.calib))
        mean = mean[cut]
        var = var[cut]
        snd = snd[cut]
        indices = indices[cut]
    except IOError:
        print('dark.npz not found')

    snd_thresh = args.snd_thresh
    mean_thresh = args.mean_thresh
    var_thresh = args.var_thresh
    
    #TODO
    if False: 
        # compute heuristically 
        print('computing threshold and finding hotcells')
        temp_mean_sm = np.mean(snd_max)
        std_snd_max = np.std(snd_max)

        # sort snd_max and find largest values in second max
        temp_large_sm  = snd_max[snd_max > temp_mean_sm]
        
        # compute thresh, make cuts, obtain hotcells
        large_sm_mean    = np.mean(temp_large_sm)
        large_sm_std     = np.std(temp_large_sm)
        
        # extra buffer for large standard deviations
        thresh = large_sm_mean + large_sm_std
    

    snd_cut   = (snd <= snd_thresh)
    mean_cut  = (mean <= mean_thresh)
    var_cut   = (var <= var_thresh)
    keep      = snd_cut & mean_cut & var_cut
    comp_mean = mean[keep]     
    comp_var  = var[keep]
    comp_snd  = snd[keep]
    hotcells  = indices[np.logical_not(keep)]

    # calculate new means and variances to plot
    print('done.\ngetting data:')
    print('\n   mean/variance array information: \n')
    print('mean size:           ', mean.size)
    print('variance size:       ', var.size)
    print('cut mean size:       ', comp_mean.size)
    print('cut variance size:   ', comp_var.size)

    print('\n   computation information \n')
    print('total resolution:       ', total_res)
    print('threshold:               %.3f' % args.snd_thresh)
    print('number of hotcells:     ', hotcells.size)
    print('%% of hotcells found:     %.3f' % (100.0 * (hotcells.size / total_res)))    

    if args.commit:
        export(hotcells, os.path.join(args.calib, 'hot_online.npz'))
    
    if args.plot:
        plot(mean, var, snd, comp_mean, comp_var, comp_snd, snd_thresh) 
    


if __name__ == '__main__':

    parser = ArgumentParser(description='')
    parser.add_argument('dark', help='.npz file for dark run')

    parser.add_argument('--calib', default='calib', help="calibration directory")
    parser.add_argument('--snd_thresh', type=int, default=1023, help='second max threshold to apply')
    parser.add_argument('--mean_thresh', type=float, default=1e9, help='mean threshold to apply')
    parser.add_argument('--var_thresh', type=float, default=1e9, help='variance threshold to apply')
    
    parser.add_argument('--commit', action='store_true', help='commit to hot_online.npz')
    parser.add_argument('--raw', action='store_true', help='Do not apply lens shading corrections')
    parser.add_argument('--plot', action="store_true", help="plot hotcell elimination results")
    parser.add_argument('--small', action='store_true', help='Use small plot size')

    args = parser.parse_args()
    data = np.load(args.dark)

    compute(data)
