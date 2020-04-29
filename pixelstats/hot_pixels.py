#!/usr/bin/env python3
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from argparse import ArgumentParser
import numpy as np
import os

from geometry import load_res
from dark_pixels import load_dark
from lens_shading import load_weights

def export(hot_cells, outfile):
    print('exporting hotcells as .npz file')
    flat_hotcell_list = hot_cells.flatten()
    np.savez(outfile, hot_list=flat_hotcell_list)
    print('done.')


def plot(mean, variance, snd_max, cut_mean, cut_variance, cut_snd_max, thresh):
    print('\nplotting figures:')
    
    plt.figure(figsize=(12,6))
    plt.subplot(121) 
    cut = np.random.randint(mean.size, size=100000)
    plt.scatter(mean[cut], variance[cut], c=snd_max[cut], s=0.5, cmap='rainbow', vmax=thresh)
    #plt.hist2d(mean, variance, bins=500, norm=LogNorm())
    plt.title('Variance vs. Mean: Before cut')
    plt.xlabel('Mean')
    plt.ylabel('Variance')
    plt.colorbar()

    plt.subplot(122)
    cut = np.random.randint(cut_mean.size, size=100000)
    plt.scatter(cut_mean[cut], cut_variance[cut], c=cut_snd_max[cut], s=0.5, cmap='rainbow', vmax=thresh)
    #plt.hist2d(cut_mean, cut_variance, bins=500, norm=LogNorm())
    plt.title('Variance vs. Mean: After cut')
    plt.xlabel('Mean')
    plt.ylabel('Variance')
    plt.colorbar()
  
    plt.show()
    print('done.')


def compute(data):
    
    sum_ = data['sum']
    num_ = data['num']
    ssq_ = data['ssq']
    snd_max = data['second']

    data.close()

    mean = sum_ / num_
    variance = (( ssq_ / num_ ) - ( (sum_ / num_)**2 )) * (( num_ / (num_ - 1) ))
    total_res = mean.size
    indices = np.arange(total_res)

    if not args.raw:
        try:
            wgt = load_weights(args.calib).flatten()
            mean *= wgt
            variance *= wgt**2
            snd_max *= wgt
        except IOError:
            print('lens.npz not found')

    try:
        cut = np.logical_not(load_dark(args.calib))
        mean = mean[cut]
        variance = variance[cut]
        snd_max = snd_max[cut]
        indices = indices[cut]
    except IOError:
        print('dark.npz not found')

    thresh = args.thresh
    if not thresh: 
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
    
    keep_pixels      = (snd_max <= thresh) & (variance < 2000) # for displaying
    comp_mean        = mean[keep_pixels]     
    comp_variance    = variance[keep_pixels]
    comp_snd_max     = snd_max[keep_pixels]
    hotcells         = indices[np.logical_not(keep_pixels)]

    # calculate new means and variances to plot
    print('done.\ngetting data:')
    print('\n   mean/variance array information: \n')
    print('mean size:           ', mean.size)
    print('variance size:       ', variance.size)
    print('cut mean size:       ', comp_mean.size)
    print('cut variance size:   ', comp_variance.size)

    print('\n   computation information \n')
    print('total resolution:       ', total_res)
    print('threshold:               %.3f' % thresh)
    print('number of hotcells:     ', hotcells.size)
    print('%% of hotcells found:     %.3f' % (100.0 * (hotcells.size / total_res)))    

    if args.commit:
        export(hotcells, os.path.join(args.calib, 'hot_online.npz'))
    
    if args.plot:
        plot(mean, variance, snd_max, comp_mean, comp_variance, comp_snd_max, thresh) 
    


if __name__ == '__main__':

    parser = ArgumentParser(description='')
    parser.add_argument('dark', help='.npz file for dark run')

    parser.add_argument('--calib', default='calib', help="calibration directory")
    parser.add_argument('--thresh', type=int, help='second max threshold to apply')
    
    parser.add_argument('--commit', action='store_true', help='commit to hot_online.npz')
    parser.add_argument('--raw', action='store_true', help='Do not apply lens shading corrections')
    parser.add_argument('--plot', action="store_true", help="plot hotcell elimination results")

    args = parser.parse_args()
    data = np.load(args.dark)

    compute(data)
