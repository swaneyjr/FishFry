#!/usr/bin/env python3


from unpack import *
from matplotlib.colors import LogNorm
from geometry import load_res
from dark_pixels import load_dark
from hot_pixels import load_hot
import matplotlib.pyplot as plt
import argparse
import sys
import os

def load_weights(calib):
    f_lens = np.load(os.path.join(calib, 'lens.npz'))
    
    gain = f_lens['lens']
    ds   = f_lens['down_sample']

    f_lens.close()

    wgt = np.where(gain > 0, 1/gain, 0)
    wgt /= wgt.max()

    return np.repeat(np.repeat(wgt, ds, axis=0), ds, axis=1)
    

def calculate(calib, rsq_thresh=0, calib_dark=True, calib_hot=True, ds=4):

    print('loading image geometry')
    width, height = load_res(calib)
    index  = np.arange(width*height,dtype=int)
    xpos = index % width
    ypos = index / width

    # load the pixel gains:
    try:
        print('loading pixel gains')
        filename = os.path.join(calib, "gain.npz")
        fgains  = np.load(filename)
    except:
        print("could not process file ", filename, " as .npz file.  Run gain.py with --commit option first?")
        return
    gain       = fgains['gain']
    intercept  = fgains['intercept'] 
    rsq        = fgains['rsq']

    fgains.close()

    # set bad pixels to NaN
    infs = np.isinf(gain) | np.isinf(intercept) | np.isnan(intercept)
    gain[infs] = np.nan

    if rsq_thresh:
        gain[rsq < rsq_thresh] = np.nan

    if calib_dark:
        try:
            print('loading dark pixels')
            dark = load_dark(calib)
            gain[dark] = np.nan
        except IOError:
            print("dark pixel file could not be processed")
     
    if calib_hot:
        try:
            print('loading hot pixels')
            hot = load_hot(calib)
            gain[hot] = np.nan
        except IOError:
            print("hot pixel file could not be processed") 

    print('beginning to downsample')

    nx = width // ds
    ny = height // ds

    # take mean of each n x n block
    gain_ds = gain.reshape(ny, ds, nx, ds)
    g_means = np.nanmedian(gain_ds, axis=(1,3))
    
    print('downsampling completed, displaying information:')
    print('nx:                ', nx)
    print('ny:                ', ny)
    print('lens.size:         ', g_means.size)
    print('lens.shape:        ', g_means.shape)
    print('NaN found:         ', np.sum(np.isnan(g_means)))
    print()
    
    print('computed lens shading attributes')

    return g_means


def load(calib):    
    flens = np.load(os.path.join(calib, 'lens.npz'))
    lens  = flens["lens"]
    flens.close()
    return lens

def plot(lens, min_gain=None, max_gain=None):
    print('plotting computed lens')
    plt.imshow(lens, vmin=min_gain, vmax=max_gain)
    plt.colorbar()
    #plt.savefig("plots/lens.pdf")
    plt.show()
     

        
if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='construct the lens shading map from calculated gains', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--calib',default='calib', help='calibration directory to use')
    parser.add_argument('--keep_dark',action="store_true", help="explicitly include dark pixels in analysis.")
    parser.add_argument('--keep_hot',action="store_true", help="explicitly include hot pixels in analysis.")
    parser.add_argument('--min_rsq',type=float,help='minimum r^2 value to be considered in downsampled gain')
    parser.add_argument('--down_sample',  metavar="DOWN", type=int, default=4,help="down sample by amount DOWN.")
    parser.add_argument('--max_gain',  type=float, help="minimum gain for plot.")
    parser.add_argument('--min_gain',  type=float, help="maximum gain for plot.")
    parser.add_argument('--plot_only',action="store_true", help="load and plot previous results.")

    parser.add_argument('--commit', action="store_true", help="commit lens.npz file")
    parser.add_argument('--plot', action="store_true", help="plot lens shading values")

    args = parser.parse_args()
    
    if args.plot_only:
        lens = load(args.calib) 
        plot(lens, args.min_gain, args.max_gain)
        
    else: 
        lens=calculate(args.calib, 
                calib_dark=(not args.keep_dark), 
                calib_hot=(not args.keep_hot),
                rsq_thresh = args.min_rsq,
                ds=args.down_sample)

        if args.commit:
            filename = os.path.join(args.calib, 'lens.npz')
            print('computed lens shading committed to ', filename)
            np.savez(filename, down_sample=args.down_sample, lens=lens)

        if args.plot: 
            plot(lens, args.min_gain, args.max_gain)




