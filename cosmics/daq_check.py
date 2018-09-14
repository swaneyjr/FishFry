#!/usr/bin/env python

import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import unpack_hist as hist
import unpack_trigger as trigger

from calibrate import *

def process_hist(filename, args):

    # load data:
    header,hist_uncal,hist_unhot,hist_calib = hist.unpack_all(filename)
    images = hist.interpret_header(header, "images")
    width  = hist.interpret_header(header, "width")
    height = hist.interpret_header(header, "height")

    norm = float(width) * float(height) * float(images)
    hist_uncal = hist_uncal.astype(float)
    hist_calib = hist_calib.astype(float)

    print hist_uncal
    print hist_calib
        
    err_uncal = hist_uncal**0.5
    err_calib = hist_calib**0.5
    
    # scale counts to a rate:
    hist_uncal = hist_uncal / norm
    hist_calib = hist_calib / norm
    err_uncal = err_uncal / norm
    err_calib = err_calib / norm

    cbins = np.arange(hist_calib.size)

    if (args.calib):
        return cbins, hist_calib, err_calib;
    else:
        return cbins, hist_uncal, err_uncal;

def process_trig(filename,args,ref_bins, ref_hist, ref_err):
    header,px,py,highest,region,images,dropped = trigger.unpack_all(filename)
    trigger.show_header(header)
    num_zerobias = trigger.interpret_header(header, "num_zerobias")
    width  = trigger.interpret_header(header, "width")
    height = trigger.interpret_header(header, "height")

    if (args.calib):
        dx = trigger.interpret_header(header,"region_dx")
        dy = trigger.interpret_header(header,"region_dy")
        region = calibrate_region(px,py,region,dx,dy)
   


    threshold,prescale = trigger.get_trigger(header)

    print "images:        ", images
    print "width:         ", width
    print "height:        ", height
    print "num_zerobias:  ", num_zerobias
    zb_wgt = 1.0 / (images * num_zerobias);


    print "zero-bias:  ", np.sum(highest==0)/images
    for i in range(prescale.size):
        print i, " prescale:  ", prescale[i], ":  ", np.sum(highest==i+1)/images

    zb = region[(highest==0),12]
    hzb,bins = np.histogram(np.clip(zb,0,200), bins=200, range=(0,200))
    errzb = hzb**0.5
    hzb = hzb * zb_wgt;
    errzb = errzb * zb_wgt;
    cbins = bins[:-1] 

    trig_hist = []
    trig_err  = []

    for i in range(prescale.size):
        ps = prescale[i]
        print "prescale:  ", ps
        trig_wgt = float(ps) / (float(images) * float(width) * float(height))
        print "trig_wgt:  ", trig_wgt;        
        trig = region[(highest==(1+i)),12]
        h,bins = np.histogram(np.clip(trig,0,200), bins=200, range=(0,200))
        err = h**0.5
        h = h * trig_wgt;
        err = err * trig_wgt;
        trig_hist.append(h)
        trig_err.append(err)



    plt.errorbar(ref_bins,ref_hist,yerr=ref_err,color="black",fmt="--")
    plt.errorbar(cbins[:5],hzb[:5],yerr=errzb[:5],color="black",fmt="o")
    for i in range(prescale.size):
        plt.errorbar(cbins,trig_hist[i],yerr=trig_err[i],fmt="o")
    plt.xlabel("pixel value")
    plt.ylabel("rate per pixel per image")
    plt.yscale('log')
    plt.xlim(0,args.max)



    plt.show()

    return 


    count = 0
    for i in range(num_region):
        h = highest[i];
        print "px:       ", px[i]
        print "py:       ", py[i]
        print "highest:  ", h
        print "region:   ", region[i]
        count += 1

    print "images:                   ", images
    print "total number of regions:  ", num_region
    print "regions shown:            ", count
    print "total dropped triggers:   ", dropped
    
if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Plot rate from Cosmics.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('hist', metavar='HIST', help='histogram file to process')
    parser.add_argument('trig', metavar='TRIG', nargs='+', help='trigger file(s) to process')
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--calib',action="store_true", help="compare calibrated pixel values.")
    parser.add_argument('--max',  type=int, default=50,help="maximum pixel value in rate plot (x-axis).")
    args = parser.parse_args()

    print "processing histogram file:  ", args.hist
    ref_bins, ref_hist, ref_err = process_hist(args.hist,args)
    for filename in args.trig:
        print "processing trigger file:  ", filename
        process_trig(filename, args, ref_bins, ref_hist, ref_err)
