#!/usr/bin/env python

import sys
from unpack_hist import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import argparse

def process(filename, args):

    # load data:
    header,hist_uncal,hist_unhot,hist_calib = unpack_all(filename)
    images = interpret_header(header, "images")
    width  = interpret_header(header, "width")
    height = interpret_header(header, "height")


    norm = float(width) * float(height) * float(images)
    rate_uncal = hist_uncal.astype(float)
    rate_unhot = hist_unhot.astype(float)
    rate_calib = hist_calib.astype(float)

    cbins = np.arange(0,hist_uncal.size) 
    err_uncal = rate_uncal**0.5
    err_unhot = rate_unhot**0.5
    err_calib = rate_calib**0.5
    
    # scale counts to a rate:
    rate_uncal = rate_uncal / norm
    rate_unhot = rate_unhot / norm
    rate_calib = rate_calib / norm
    err_uncal = err_uncal / norm
    err_unhot = err_unhot / norm
    err_calib = err_calib / norm

    if (args.inclusive):
        rate_inc = np.array([np.sum(hist_calib[i:]) for i in range(hist_calib.size)])
        rate_err = rate_inc**0.5
        rate_inc = rate_inc.astype(float) / images
        rate_err = rate_err.astype(float) / images
        plt.errorbar(cbins,rate_inc,yerr=rate_err,color="black",fmt="o")
        plt.xlim(0,args.max)
        plt.yscale('log')
        plt.show()
        
        prescale = 1;
        for i in np.arange(rate_inc.size,0,-1)-1:
            while (rate_inc[i] >= args.num*prescale):
                print "prescale: ", prescale, "threshold: ", i
                prescale *= 8
        return

    
    plt.errorbar(cbins,hist_uncal,yerr=err_uncal,color="black",fmt="o")
    plt.errorbar(cbins,hist_unhot,yerr=err_unhot,color="red",fmt="o")
    plt.errorbar(cbins,hist_calib,yerr=err_calib,color="blue",fmt="o")

    plt.xlabel("pixel value")
    plt.ylabel("rate per pixel per image")
    plt.yscale('log')
    plt.xlim(0,args.max)
    plt.savefig("plots/rate.pdf")
    plt.show()
    
    
    
if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Plot rate from Cosmics.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--skip_default',action="store_true", help="skip the default plot or plots.")
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--max',  type=int, default=200,help="maximum pixel value in rate plot (x-axis).")
    parser.add_argument('--inclusive',action="store_true", help="plot inclusive rate at each threshold")
    parser.add_argument('--num',  metavar='NUM', type=int, default=10,help="find prescales and thresholds yielding NUM pixels per event.")
    args = parser.parse_args()

    for filename in args.files:
        print "processing file:  ", filename
        process(filename, args)




