#!/usr/bin/env python

import sys
from unpack import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import argparse

from geometry import *


def analysis(args):

    # load the image geometry from the calibrations:
    try:
        geom = np.load("calib/geometry.npz");
    except:
        print "calib/geometry.npz does not exist.  Use dump_header.py --geometry"
        return
    width  = geom["width"]
    height = geom["height"]
    index  = np.arange(width*height,dtype=int)
    xpos = index % width
    ypos = index / width

    # load the dark pixel map:
    #try:
    #        all_dark = np.load("calib/all_dark.npy")            
    #except:
    #        print "dark pixel file calib/all_dark.npy does not exist."
    #        return

    try:
        filename = "calib/gain.npz"
        gains  = np.load(filename);
    except:
        print "could not process file ", filename, " as .npz file.  Run gain.py with --commit option first?"
        return

    try:
        filename = "calib/gain_points.npz"
        points  = np.load(filename);
    except:
        print "could not process file ", filename, " as .npz file.  Run gain.py with --commit option first?"
        return

    print gains.files
    print points.files

    gain       = gains['gain']
    intercept  = gains['intercept']
    count      = points['count']
    var        = points['vars']
    mean       = points['means']

    x = np.array([gain.size, intercept.size, count.size])
    if (not np.all(x == width*height)):
        print "ERROR: inconsistent file size... I'm out!"
        return
                 
    x = np.array([var.size, mean.size])
    if (not np.all(x == np.sum(count))):
        print "ERROR: inconsistent number of data points... I'm out!"
        return

    good = np.isfinite(gain) & np.isfinite(intercept) 
    gain[(good == False)] = -1
    intercept[(good == False)] = 0
    good = good & (gain>0)

    print "failed fits:  ", np.sum(good == False)
    normal = good #& (all_dark == False)
    #dark   = good & (all_dark == True)

    # variance with gain factor removed:
    evar = intercept/np.square(gain)
    
    print "normal pixels:  ";
    print "average gain:   ", np.mean(gain[normal])
    print "average intercept:  ", np.mean(intercept[normal])
    print "average electron equivalent variance:  ", np.mean(evar[normal]) 

    #print "dark pixels:  ";
    #print "average gain:   ", np.mean(gain[dark])
    #print "average intercept:  ", np.mean(intercept[dark])

    if (0):
        hn,bins = np.histogram(np.clip(evar[normal],-10,100), bins=50, range=(-10,100))
        errn = hn**0.5
        cbins = 0.5*(bins[:-1] + bins[1:])
        plt.errorbar(cbins,hn,yerr=errn,color="blue",fmt="o")
        plt.yscale('log')
        plt.ylabel("pixels")
        plt.xlabel("electron variance")
        plt.show()

    if (0):
        hn,bins = np.histogram(np.clip(gain[normal],0,20), bins=100, range=(0,20))
        errn = hn**0.5
        #hd,bins = np.histogram(np.clip(gain[dark],0,20), bins=100, range=(0,20))
        #errd = hd**0.5
        cbins = 0.5*(bins[:-1] + bins[1:])
        plt.errorbar(cbins,hn,yerr=errn,color="blue",fmt="o")
        plt.errorbar(cbins,hd,yerr=errd,color="red",fmt="o")
        plt.yscale('log')
        plt.ylabel("pixels")
        plt.xlabel("gain (ADC/e)")
        plt.show()


    # position for down sampling by 8x8 
    down, nx, ny = down_sample(width, height, 8, 8)
    G = np.zeros(nx*ny, dtype=float)
    S = np.zeros(nx*ny, dtype=float)
    for i in np.where(normal):
        G[down[i]] += gain[i]
        S[down[i]] += 1.0
    nz = np.nonzero(S)[0]
    G[nz] = G[nz]/S[nz]


    avg_gain = np.array([G[down[i]] for i in range(width*height)])
    diff = gain - avg_gain
    print "mean discrepancy:  ", np.mean(diff[normal])
    print "rms discrepancy:   ", np.sqrt(np.var(diff[normal]))
    keep = normal & (np.abs(diff) < 2.0) 
    
    if (1):
        G = G.reshape(ny,nx)
        plt.imshow(G, vmin=0, vmax=5)
        plt.colorbar()
        plt.show()
        G = G.reshape(ny*nx)

    hn,bins = np.histogram(diff[normal], bins=100, range=(-10,10))
    errn = hn**0.5
    cbins = 0.5*(bins[:-1] + bins[1:])
    plt.errorbar(cbins,hn,yerr=errn,color="blue",fmt="o")
    plt.yscale('log')
    plt.ylabel("pixels")
    plt.xlabel("difference with neighbors")
    plt.show()

    print "mean discrepancy:  ", np.mean(diff[keep])
    print "rms discrepancy:   ", np.sqrt(np.var(diff[keep]))

    print "saving average gain results"
    np.savez("calib/avg_gain.npz", gain=avg_gain, keep=keep)
    
    return
    
    
if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Plot results of pixel gain fit', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--max_var',  type=float, default=800,help="input files have not been preprocessed.")
    parser.add_argument('--max_mean', type=float, default=200,help="input files have not been preprocessed.")
    parser.add_argument('--by_filter',action="store_true", help="produce 4 plots for each corner of the 2x2 filter arrangement.")
    args = parser.parse_args()

    analysis(args)




