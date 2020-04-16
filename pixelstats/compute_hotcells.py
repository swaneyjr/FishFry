#!/usr/bin/env python3
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from argparse import ArgumentParser
import numpy as np


def export(hot_cells):
    print('exporting hotcells as .npz file')
    flat_hotcell_list = hot_cells.flatten()
    np.savez('calib/hot.npz', hot_list=flat_hotcell_list)
    print('done.')


def plot(mean, variance, cut_mean, cut_variance):
    print('\nplotting figures:')
    plt.figure(1)
    plt.hist2d(mean, variance, bins=500, norm=LogNorm())
    plt.title('Variance vs. Mean: Before cut')
    plt.xlabel('Mean')
    plt.ylabel('Variance')
    plt.colorbar()

    plt.figure(2)
    plt.hist2d(cut_mean, cut_variance, bins=500, norm=LogNorm())
    plt.title('Variance vs. Mean: After cut')
    plt.xlabel('Mean')
    plt.ylabel('Variance')
    plt.colorbar()
  
    plt.show()
    print('done.')


def compute(data):

    '''  
    Phone resolutions
    S6: (5238, 3000)
    V20: (4640, 3480)
    '''
    x_res = 4640
    y_res = 3480
    total_res = x_res * y_res
    
    sum_ = data['sum']
    num_ = data['num']
    ssq_ = data['ssq']
    snd_max = data['second']

    mean = sum_ / num_
    variance = (( ssq_ / num_ ) - ( (sum_ / num_)**2 )) * (( num_ / (num_ - 1) ))
    std_snd_max = np.std(snd_max)

    print('computing threshold and finding hotcells')
    temp_mean_sm = np.mean(snd_max)
    
    # sort snd_max and find largest values in second max
    sorted_snd_max = np.sort(snd_max)[::-1]
    temp_large_sm  = []
    for i in sorted_snd_max:
        if (i > temp_mean_sm):
           temp_large_sm.append(i)
    
    # compute thresh, make cuts, obtain hotcells
    large_sm_log     = np.array(temp_large_sm)
    large_sm_mean    = np.mean(large_sm_log)
    large_sm_std     = np.std(temp_large_sm)
    
    # extra buffer for large standard deviations
    thresh = large_sm_mean if large_sm_std > 50 else large_sm_mean + large_sm_std
    
    keep_pixels      = (snd_max <= thresh) & (variance < 2000) # for displaying
    comp_mean        = mean[keep_pixels]     
    comp_variance    = variance[keep_pixels]
   
    hotcells = np.argwhere(np.logical_not(keep_pixels))

    # calculate new means and variances to plot
    print('done.\ngetting data:')
    print('\n   mean/variance array information: \n')
    print('mean size:           ', mean.size)
    print('variance size:       ', variance.size)
    print('cut mean size:       ', comp_mean.size)
    print('cut variance size:   ', comp_variance.size)

    print('\n   computation information \n')
    print('total resolution:       ', total_res)
    print('large snd_max mean:      %.3f' % large_sm_mean)
    print('large snd_max std:       %.3f' % large_sm_std)
    print('threshold:               %.3f' % thresh)
    print('number of hotcells:     ', hotcells.size)
    print('%% of hotcells found:     %.3f' % (100.0 * (hotcells.size / total_res)))    

    if args.commit:
        export(hotcells)
    
    if args.plot:
        plot(mean, variance, comp_mean, comp_variance) 
    


if __name__ == '__main__':

    parser = ArgumentParser(description='')
    parser.add_argument('npz', help='.npz file to load')

    parser.add_argument('--commit',action="store_true", help="commit hotcells as hot.npz")
    parser.add_argument('--plot',action="store_true", help="plot hotcell elimination results")
    parser.add_argument('--lens', action="store_true", help="compute hotcells that have lens shading")

    args = parser.parse_args()
    npz = np.load(args.npz)

    compute(npz)
