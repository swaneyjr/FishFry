#!/usr/bin/env python3


from unpack import *
from matplotlib.colors import LogNorm
from geometry import load_res, down_sample
import matplotlib.pyplot as plt
import argparse
import sys

FILE_NAME = "calib/lens.npz"

def calculate(args):

    print('loading image geometry')
    width, height = load_res()
    index  = np.arange(width*height,dtype=int)
    xpos = index % width
    ypos = index / width

    # load the pixel gains:
    try:
        filename = "calib/gain.npz"
        gains  = np.load(filename);
    except:
        print("could not process file ", filename, " as .npz file.  Run gain.py with --commit option first?")
        return
    gain       = gains['gain']
    intercept  = gains['intercept']

    # boolean array of 15984000
    keep = np.isfinite(gain) & np.isfinite(intercept)

    if (args.no_dark):
        try:
            all_dark = np.load("calib/all_dark.npy")            
        except:
            print("dark pixel file calib/all_dark.npy does not exist.")
            return
        keep = keep & (all_dark == False)

    if (args.no_hot):
        try:
            filename = "calib/hot.npz"
            hots  = np.load(filename);
        except:
            print("could not process file ", filename, " as .npz file.")
            return        
        hot = hots['hot_list']
        all_hot = np.full(width*height, False)
        all_hot[hot] = True;
        keep = keep & (all_hot == False)

    print('beginning to downsample')
    n       = 4               # used for creating an n x n block
    G       = np.array([])    # reshapes original gain array using n
    arr     = np.array([])    # further reshapes array down for downsampling
    means   = np.array([])    # means of 4x4 blocks of pixels, stored in each element
    

    # based on proportionality constant of width / height = 1.776
    down, nx, ny = down_sample(width, height, n, n)
    nx           = np.int(nx)
    ny           = np.int(ny)
    G            = gain.reshape( int(gain.shape[0] // (n**2)), int(n**2) )
    
    if args.s6:
        arr = G.reshape( 3000 // n, n, 5328 // n, n)
    
    if args.v20:
        arr = G.reshape( 4640 // n, n, 3480 // n, n)

    means        = arr.mean(axis=(1,3))
    
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
    print('G.size:            ', G.size)
    print('G.shape:           ', G.shape)
    print('arr.shape:         ', arr.shape)
    print('arr.size:          ', arr.size)
    print('means.shape:       ', means.shape)
    print('means.size:        ', means.size)
    print()

    # map back to full resolution need for applying to gain values
    # -> not technically 'full resolution'
    lens = means.reshape(ny, nx)
    
    print('computed lens shading attributes')
    attributes('lens', lens) 
     
      
    if args.commit:
        print('computed lens shading committed to ', FILE_NAME)
        np.savez(FILE_NAME, down_sample=n, lens=lens)

    if args.expt:
        print('displaying experimental plots')
        plt.figure(1)
        plt.plot(means, gain)
        plt.xlabel('mean')
        plt.ylabel('variance')


    return lens

def load(args):    
    try:
        lens = np.load(FILE_NAME);
    except:
        print("calib/lens.npz does not exist.  Run without --plot_only first?")
        return
    lens  = lens["lens"]
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
    parser.add_argument('--no_dark',action="store_true", help="exclude dark pixels from analysis.")
    parser.add_argument('--no_hot',action="store_true", help="exclude hot pixels from analysis.")
    parser.add_argument('--short',action="store_true", help="short (test) analysis of a few pixels.")
    parser.add_argument('--down_sample',  metavar="DOWN", type=int, default=8,help="down sample by amount DOWN.")
    parser.add_argument('--max_gain',  type=float, default=10,help="minimum gain for plot.")
    parser.add_argument('--min_gain',  type=float, default=0,help="maximum gain for plot.")
    parser.add_argument('--plot_only',action="store_true", help="load and plot previous results.")

    parser.add_argument('--commit', action="store_true", help="commit lens.npz file")
    parser.add_argument('--plot', action="store_true", help="plot lens shading values")
    parser.add_argument('--rsquared', action="store_true", help="plot R^2 values")
    parser.add_argument('--expt', action="store_true", help="plot experimental plots")
    
    parser.add_argument('-s6', action="store_true", help="work in S6 resolution")
    parser.add_argument('-v20', action="store_true", help="work in V20 resolution")

    args = parser.parse_args()
    analysis(args)




