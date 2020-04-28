#!/usr/bin/env python3

import sys
import os
from unpack import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from geometry import load_res
from dark_pixels import load_dark

import argparse

def process(filename, args):

    # load data, from either raw file directly from phone, or as output from combine.py utility:

    if args.raw:
         version,header,sum,ssq = unpack_all(filename)
         index = get_pixel_indices(header)
         images = interpret_header(header, "images")
         num = np.full(index.size, images)
         width  = interpret_header(header, "width")
         height = interpret_header(header, "height")
    else:
        # load the image geometry from the calibrations:
        width, height = load_res(args.calib)

        try:
            npz = np.load(filename)
        except:
            print("could not process file ", filename, " as .npz file.  Use --raw option?")
            return

        sum      = npz['sum']
        ssq      = npz['ssq']
        num     = npz['num'] 
    
    cmean = sum / num
    cvari = (ssq / num - cmean**2) * num / (num-1)

    # first show spatial distribution
    plt.figure(1, figsize=(6,8))
    plt.subplot(211)
    plt.imshow(cmean.reshape(height, width), norm=LogNorm(), cmap='coolwarm')
    plt.colorbar()

    plt.subplot(212)
    plt.imshow(cvari.reshape(height, width), norm=LogNorm(), cmap='coolwarm')
    plt.colorbar()

    # now do 2D histogram(s) for mean and variance 
    plt.figure(2, figsize=(10,8))

    # first apply gains if appropriate
    if args.gain:
        try:
            filename = os.path.join(args.calib, 'lens.npz')
            lens = np.load(filename)            
        except:
            print("average gain file", filename, "does not exist.")
            return
        avg_gain = lens['avg_gain']
        gain = np.array([avg_gain[i] for i in index])
        cmean = cmean / gain
        cvari = cvari / np.square(gain) 


    # now select pixels to plot
    index = np.arange(sum.size)
    xpos = index % width
    ypos = index // width
    keep = np.ones(width*height, dtype=bool)

    if args.no_dark or args.all_dark:
        try:
            dark = load_dark(args.calib)
        except:
            print("dark pixel file", filename, "does not exist.")
            return
        
        if args.no_dark:
            keep &= np.logical_not(dark)
        if args.all_dark:
            keep &= dark

    if args.hot:
        max_mean = args.hot[0]
        max_vari  = args.hot[1]
        print("saving hot pixel list from mean > ", max_mean, " or var > ", max_vari)
        hot = ((cmean > max_mean) + (cvari > max_vari))
        hot_list = index[hot]
        hotfile = os.path.join(args.calib, "hot_online.npz")
        print("saving  ", hot_list.size, "hot pixels to file ", hotfile)
        print("total pixels in device:  ", width * height)
        frac = hot_list.size / (width*height)
        print("faction of hot pixels:  ", frac)

        np.savez(hotfile, hot_list=hot_list)

        keep &= (hot == False)
         

    if args.by_filter or args.by_radius:

        # 4 subplots 

        if args.by_filter:
            posA = keep * ((xpos%2)==0) * ((ypos%2)==0)
            posB = keep * ((xpos%2)==1) * ((ypos%2)==0)
            posC = keep * ((xpos%2)==0) * ((ypos%2)==1)
            posD = keep * ((xpos%2)==1) * ((ypos%2)==1)
        else:
            rs     = ((xpos - width/2.0)**2 + (ypos - height/2.0)**2)
            norm   = (width/2.0)**2 + (height/2.0)**2
            rs     = rs/norm

            posA = keep * (rs >= 0.75)
            posB = keep * (rs < 0.75) * (rs >= 0.5) 
            posC = keep * (rs < 0.5) * (rs >= 0.25) 
            posD = keep * (rs < 0.25) 


        kwargs = {
                'norm': LogNorm(),
                'bins': [500, 500],
                'range':[[0,args.max_mean],[0,args.max_var]],
                } 
 
        plt.subplot(221)
        plt.hist2d(cmean[posA],cvari[posA],**kwargs)
        plt.xlabel("mean")
        plt.ylabel("variance")    
        plt.subplot(222)
        plt.hist2d(cmean[posB],cvari[posB],**kwargs)
        plt.xlabel("mean")
        plt.ylabel("variance")    
        plt.subplot(223)
        plt.hist2d(cmean[posC],cvari[posC],**kwargs)
        plt.xlabel("mean")
        plt.ylabel("variance")    
        plt.subplot(224)
        plt.hist2d(cmean[posD],cvari[posD],**kwargs)
        plt.xlabel("mean")
        plt.ylabel("variance")    
        plt.show() 

    else:
        plt.hist2d(cmean,cvari,norm=LogNorm(),bins=[500,500],range=[[0,args.max_mean],[0,args.max_var]])
        plt.xlabel("mean")
        plt.ylabel("variance")
        

    if args.save_plot:
        plot_name = "plots/mean_var_calib.pdf" if args.calib \
                else "plots/mean_var.pdf"
        print("saving plot to file:  ", plot_name)
        plt.savefig(plot_name)
        
    plt.show()

    return

    h,bins = np.histogram(np.clip(cmean,0,10), bins=100, range=(0,10))
    err = h**0.5
    cbins = 0.5*(bins[:-1] + bins[1:])
    plt.errorbar(cbins,h,yerr=err,color="black",fmt="o")
    plt.ylim(1.0,1E7)
    plt.ylabel("pixels")
    plt.xlabel("mean")
    plt.yscale('log')
    plt.savefig("hmean.pdf")
    plt.show()
    
    
    h,bins = np.histogram(np.clip(cvari,0,100), bins=100, range=(0,100))
    err = h**0.5
    cbins = 0.5*(bins[:-1] + bins[1:])
    plt.errorbar(cbins,h,yerr=err,color="black",fmt="o")
    plt.ylim(1.0,1E7)
    plt.ylabel("pixels")
    plt.xlabel("variance")
    plt.yscale('log')
    plt.savefig("hvari.pdf")
    plt.show()


    
if __name__ == "__main__":
    example_text = '''examples:

    ./show_mean_var.py ./data/combined/small_dark.npz --max_var=30 --max_mean=5'''
    
    parser = argparse.ArgumentParser(description='Plot mean and variance.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--raw',action="store_true", help="input files have not been preprocessed.")
    parser.add_argument('--max_var',  type=float, default=800,help="input files have not been preprocessed.")
    parser.add_argument('--max_mean', type=float, default=200,help="input files have not been preprocessed.")
    parser.add_argument('--no_dark',action="store_true", help="drop dark pixels from all plots.")
    parser.add_argument('--all_dark',action="store_true", help="drop non-dark pixels from all plots.")
    parser.add_argument('--by_filter',action="store_true", help="produce 4 plots for each corner of the 2x2 filter arrangement.")
    parser.add_argument('--by_radius',action="store_true", help="produce 4 plots at three different locations from radius.")
    parser.add_argument('--gain',action="store_true", help="apply gain correction.")
    parser.add_argument('--hot', nargs=2, metavar=("MEAN","VAR"), type=float,help="save list of pixels where mean > MEAN or var > VAR")
    parser.add_argument('--calib', default='calib', help='directory with calibration files')
    parser.add_argument('--save_plot', action='store_true', help='Save plots in ./plots/ directory')
    args = parser.parse_args()

    for filename in args.files:
        print("processing file:  ", filename)
        process(filename, args)




