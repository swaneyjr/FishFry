#!/usr/bin/env python3


from unpack import *
from matplotlib.colors import LogNorm
from geometry import load_res, down_sample
import matplotlib.pyplot as plt
import argparse
import sys
import os

def calculate(args):

    print('loading image geometry')
    width, height = load_res(args.calib)
    index  = np.arange(width*height,dtype=int)
    xpos = index % width
    ypos = index / width

    # load the pixel gains:
    try:
        filename = os.path.join(args.calib, "gain.npz")
        gains  = np.load(filename);
    except:
        print("could not process file ", filename, " as .npz file.  Run gain.py with --commit option first?")
        return
    gain       = gains['gain']
    intercept  = gains['intercept']

    # boolean array of 15984000
    keep = np.isfinite(gain) & np.isfinite(intercept)

    if not args.keep_dark:
        try:
            filename = os.path.join(args.calib, 'all_dark.npy')
            all_dark = np.load(filename)           
        except:
            print("dark pixel file", filename, "does not exist.")
        keep = keep & (all_dark == False)

    if not args.no_hot:
        try:
            filename = os.path.join(args.calib, "hot_online.npz")
            hots  = np.load(filename);
        except:
            print("could not process file ", filename, " as .npz file.")
        hot = hots['hot_list']
        all_hot = np.full(width*height, False)
        all_hot[hot] = True;
        keep = keep & (all_hot == False)

    print('beginning to downsample')
    n = 4               # used for creating an n x n block
    
    res_x, res_y = load_res(args.calib)
    nx = res_x // n
    ny = res_y // n

    # take mean of each n x n block
    g_means = gain.reshape(ny, n, nx, n).mean(axis=(1,3))
    
    print('downsampling completed, displaying information:')
    print('nx:                ', nx)
    print('ny:                ', ny)
    print('nx*ny:             ', nx*ny)
    print('gain:              ', gain)
    print('gain.size:         ', gain.size)
    print('gain.shape:        ', gain.shape)
    print('intercept:         ', intercept)
    print('intercept.size:    ', intercept.size)
    print('intercept.shape:   ', intercept.shape)
    print('means.shape:       ', g_means.shape)
    print('means.size:        ', g_means.size)
    print()
    
    print('computed lens shading attributes')
      
    if args.commit:
        filename = os.path.join(args.calib, 'lens.npz')
        print('computed lens shading committed to ', filename)
        np.savez(filename, down_sample=n, lens=g_means)

    if args.expt:
        print('displaying experimental plots')
        plt.figure(1)
        plt.plot(g_means, gain)
        plt.xlabel('mean')
        plt.ylabel('variance')


    return lens

def load(args):    
    flens = np.load(os.path.join(args.calib, 'lens.npz'))
    lens  = flens["lens"]
    flens.close()
    return lens

def plot(args, lens):
    print('plotting computed lens')
    plt.imshow(lens, vmin=args.min_gain, vmax=args.max_gain)
    plt.colorbar()
    plt.savefig("plots/lens.pdf")
    plt.show()
    

def analysis(args):
    if (args.plot_only):
        lens = load(args)
    else:    
        lens = calculate(args)

    if args.plot:
        plot(args, lens)

        
if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='construct the lens shading map from calculated gains', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--calib',default='calib', help='calibration directory to use')
    parser.add_argument('--keep_dark',action="store_true", help="explicitly include dark pixels in analysis.")
    parser.add_argument('--keep_hot',action="store_true", help="explicitly include hot pixels in analysis.")
    parser.add_argument('--short',action="store_true", help="short (test) analysis of a few pixels.")
    parser.add_argument('--down_sample',  metavar="DOWN", type=int, default=8,help="down sample by amount DOWN.")
    parser.add_argument('--max_gain',  type=float, default=10,help="minimum gain for plot.")
    parser.add_argument('--min_gain',  type=float, default=0,help="maximum gain for plot.")
    parser.add_argument('--plot_only',action="store_true", help="load and plot previous results.")

    parser.add_argument('--commit', action="store_true", help="commit lens.npz file")
    parser.add_argument('--plot', action="store_true", help="plot lens shading values")
    parser.add_argument('--rsquared', action="store_true", help="plot R^2 values")
    parser.add_argument('--expt', action="store_true", help="plot experimental plots")

    args = parser.parse_args()
    analysis(args)




