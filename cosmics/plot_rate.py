#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from unpack_hist import unpack_all, interpret_header

import argparse


def plot(hist, norm=0, cum=False, ax=None, **kwargs): 

    cbins = np.arange(hist.size)
    rate = hist.astype(float)

    if cum:
        cbins = cbins[1:]
        rate = np.sum(rate) - np.cumsum(rate)[:-1]

    err = np.sqrt(rate)

    # scale counts to a rate:
    if norm:
        rate /= norm
        err /= norm
        plt.ylabel("rate per image")
    else:
        plt.ylabel('counts') 

    if ax:
        ax.errorbar(cbins,rate,yerr=err,fmt="o", **kwargs)
    else:
        plt.errorbar(cbins,rate,yerr=err,fmt="o", **kwargs)

    #plt.savefig("plots/rate.pdf")
    

def calibrate_thresholds(rate, n_trig):
    cum_rate = np.sum(rate) - np.cumsum(rate)[:-1]
    prescale = 1
    for i in np.arange(cum_rate.size,0,-1)-1:
        while cum_rate[i] >= n_trig*prescale:
            print("prescale: ", prescale, "threshold: ", i)
            prescale *= 8
    


if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Plot rate from Cosmics.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', help='files to process')
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--max',  type=int, default=200,help="maximum pixel value in rate plot (x-axis).")
    parser.add_argument('--cum',action="store_true", help="plot cumulative rate at each threshold")
    parser.add_argument('--n_trig',  metavar='NUM', type=int, default=10,help="find prescales and thresholds yielding NUM pixels per event.")
    args = parser.parse_args()

    hist_cln = 0
    hist_hot = 0
    hist_wgt = 0
    images = 0

    for filename in args.files:
        print("processing file:  ", filename)
        
        # load data:
        header, cln, hot, wgt = unpack_all(filename)

        hist_cln += cln
        hist_hot += hot
        hist_wgt += wgt

        tot_images = interpret_header(header, "images")
        prescale   = interpret_header(header, 'hist_prescale')

        images += tot_images / prescale

    images = int(images)

    ax = plt.gca()
    hist_raw = hist_cln + hist_hot
    plot(hist_raw, norm=images, cum=args.cum, ax=ax, color="black", label='Raw')
    plot(hist_cln, norm=images, cum=args.cum, ax=ax, color='blue', label='Clean')
    plot(hist_wgt, norm=images, cum=args.cum, ax=ax, color='green', label='Calib')
    
    plt.xlabel("pixel value")
    plt.semilogy()
    plt.xlim(0,args.max)

    plt.legend()
    plt.show()

    if args.cum:
        calibrate_thresholds(hist_wgt/images, args.n_trig)

