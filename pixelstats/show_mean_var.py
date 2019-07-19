#!/usr/bin/env python

import sys
from unpack import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from geometry import load_res

import argparse

def process(filename, args):
    # plots to show:
    meanvar = True
    mvzoom  = True
    hmean   = True
    hvari  = True

    # load data, from either raw file directly from phone, or as output from combine.py utility:

    if (args.raw):
         version,header,sum,ssq = unpack_all(filename)
         index = get_pixel_indices(header)
         images = interpret_header(header, "images")
         num = np.full(index.size, images)
         width  = interpret_header(header, "width")
         height = interpret_header(header, "height")
    else:
        # load the image geometry from the calibrations:
        width, height = load_res()

        try:
            npz = np.load(filename)
        except:
            print "could not process file ", filename, " as .npz file.  Use --raw option?"
            return

        exposure = npz['exposure']
        sens     = npz['sens']
        sum      = npz['sum']
        ssq      = npz['ssq']
        num     = npz['num']
        index = np.arange(sum.size)

    
    empty = (num == 0)
    num[empty]=1
    cmean = sum / num
    cvari = ssq / num - cmean**2

    xpos = index % width
    ypos = index / width


    keep = np.full(num.size, True, dtype=bool)

    if (args.no_dark or args.all_dark):
        try:
            all_dark = np.load("calib/all_dark.npy")            
        except:
            print "dark pixel file calib/all_dark.npy does not exist."
            return

        new_dark = [all_dark[i] for i in index]
        print new_dark[:10]
        print len(new_dark)
        print num.size

        all_dark = np.array(new_dark)
        
        if (args.no_dark):
            keep = keep * (all_dark == False)
        if (args.all_dark):
            keep = keep * (all_dark == True)

    if (args.filter >= 0):
        if (args.filter == 0):
            keep = keep * ((xpos%2)==0) * ((ypos%2)==0)
        if (args.filter == 1):
            keep = keep * ((xpos%2)==1) * ((ypos%2)==0)
        if (args.filter == 2):
            keep = keep * ((xpos%2)==0) * ((ypos%2)==1)
        if (args.filter == 3):
            keep = keep * ((xpos%2)==1) * ((ypos%2)==1)

    if (args.gain):
        try:
            lens = np.load("calib/lens.npz")            
        except:
            print "average gain file calib/avg_gain.npz does not exist."
            return
        avg_gain = lens['avg_gain']
        gain = np.array([avg_gain[i] for i in index])
        cmean = cmean / gain
        cvari = cvari / np.square(gain)
        

    cmean = cmean[keep]
    cvari = cvari[keep]
    index = index[keep]
    xpos  = xpos[keep]
    ypos  = ypos[keep]

    if (args.sandbox):
        print "running development code."
        return

    if (args.hot != None):
        max_mean = args.hot[0]
        max_vari  = args.hot[1]
        print "saving hot pixel list from mean > ", max_mean, " or var > ", max_vari;
        hot = ((cmean > max_mean) + (cvari > max_vari))
        hot_list = index[hot]
        HOT_FILE = "calib/hot.npz";
        print "saving  ", hot_list.size, "hot pixels to file ", HOT_FILE;
        print "total pixels in device:  ", width * height
        frac = float(hot_list.size) / float(width*height)
        print "faction of hot pixels:  ", frac

        np.savez(HOT_FILE, hot_list=hot_list)

        keep = (hot == False)
        cmean = cmean[keep]
        cvari = cvari[keep]
        index = index[keep]
        xpos  = xpos[keep]
        ypos  = ypos[keep]
        



    if (not args.skip_default):
        plot_name = "plots/mean_var.pdf"
        if (args.gain):
            plot_name = "plots/mean_var_calib.pdf"

        plt.hist2d(cmean,cvari,norm=LogNorm(),bins=[500,500],range=[[0,args.max_mean],[0,args.max_var]])
        plt.xlabel("mean")
        plt.ylabel("variance")
        print "saving plot to file:  ", plot_name
        plt.savefig(plot_name)
        plt.show()


    if (args.by_filter):
        posA = ((xpos%2)==0) * ((ypos%2)==0)
        posB = ((xpos%2)==1) * ((ypos%2)==0)
        posC = ((xpos%2)==0) * ((ypos%2)==1)
        posD = ((xpos%2)==1) * ((ypos%2)==1)

        #posA = ((ypos%4)==0) 
        #posB = ((ypos%4)==1) 
        #posC = ((ypos%4)==2) 
        #posD = ((ypos%4)==3) 


        plt.subplot(2,2,1)
        plt.hist2d(cmean[posA],cvari[posA],norm=LogNorm(),bins=[500,500],range=[[0,args.max_mean],[0,args.max_var]])
        plt.xlabel("mean")
        plt.ylabel("variance")    
        plt.subplot(2,2,2)
        plt.hist2d(cmean[posB],cvari[posB],norm=LogNorm(),bins=[500,500],range=[[0,args.max_mean],[0,args.max_var]])
        plt.xlabel("mean")
        plt.ylabel("variance")    
        plt.subplot(2,2,3)
        plt.hist2d(cmean[posC],cvari[posC],norm=LogNorm(),bins=[500,500],range=[[0,args.max_mean],[0,args.max_var]])
        plt.xlabel("mean")
        plt.ylabel("variance")    
        plt.subplot(2,2,4)
        plt.hist2d(cmean[posD],cvari[posD],norm=LogNorm(),bins=[500,500],range=[[0,args.max_mean],[0,args.max_var]])
        plt.xlabel("mean")
        plt.ylabel("variance")    
        plt.show()

    rs     = ((xpos - width/2.0)**2 + (ypos - height/2.0)**2)
    norm   = (width/2.0)**2 + (height/2.0)**2
    rs     = rs/norm

    if (args.by_radius):
        posA = (rs >= 0.75)
        posB = (rs < 0.75) * (rs >= 0.5) 
        posC = (rs < 0.5) * (rs >= 0.25) 
        posD = (rs < 0.25) 

        plt.subplot(2,2,1)
        plt.hist2d(cmean[posA],cvari[posA],norm=LogNorm(),bins=[500,500],range=[[0,args.max_mean],[0,args.max_var]])
        plt.xlabel("mean")
        plt.ylabel("variance")    
        plt.subplot(2,2,2)
        plt.hist2d(cmean[posB],cvari[posB],norm=LogNorm(),bins=[500,500],range=[[0,args.max_mean],[0,args.max_var]])
        plt.xlabel("mean")
        plt.ylabel("variance")    
        plt.subplot(2,2,3)
        plt.hist2d(cmean[posC],cvari[posC],norm=LogNorm(),bins=[500,500],range=[[0,args.max_mean],[0,args.max_var]])
        plt.xlabel("mean")
        plt.ylabel("variance")    
        plt.subplot(2,2,4)
        plt.hist2d(cmean[posD],cvari[posD],norm=LogNorm(),bins=[500,500],range=[[0,args.max_mean],[0,args.max_var]])
        plt.xlabel("mean")
        plt.ylabel("variance")    
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
    parser.add_argument('--skip_default',action="store_true", help="skip the default plot or plots.")
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--raw',action="store_true", help="input files have not been preprocessed.")
    parser.add_argument('--max_var',  type=float, default=800,help="input files have not been preprocessed.")
    parser.add_argument('--max_mean', type=float, default=200,help="input files have not been preprocessed.")
    parser.add_argument('--no_dark',action="store_true", help="drop dark pixels from all plots.")
    parser.add_argument('--filter', metavar='POS', type=int, default=-1,help="only include pixels at filter position POS.")
    parser.add_argument('--all_dark',action="store_true", help="drop non-dark pixels from all plots.")
    parser.add_argument('--by_filter',action="store_true", help="produce 4 plots for each corner of the 2x2 filter arrangement.")
    parser.add_argument('--by_radius',action="store_true", help="produce 4 plots at three different locations from radius.")
    parser.add_argument('--gain',action="store_true", help="apply gain correction.")
    parser.add_argument('--hot', nargs=2, metavar=("MEAN","VAR"), type=float,help="save list of pixels where mean > MEAN or var > VAR")
    args = parser.parse_args()

    for filename in args.files:
        print "processing file:  ", filename
        process(filename, args)




