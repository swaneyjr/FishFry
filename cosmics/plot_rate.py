#!/usr/bin/env python

import sys
from unpack import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import argparse

def process(filename, args):

    # load data:
    header,hist_uncal,hist_calib = unpack_all(filename)
    images = interpret_header(header, "images")
    width  = interpret_header(header, "width")
    height = interpret_header(header, "height")


    norm = float(width) * float(height) * float(images)
    print width, height, images, norm

    hist_uncal = hist_uncal.astype(float)
    hist_calib = hist_calib.astype(float)
    cbins = np.arange(0,hist_uncal.size) + 0.5
    err_uncal = hist_uncal**0.5
    err_calib = hist_calib**0.5
    
    # scale counts to a rate:
    hist_uncal = hist_uncal / norm
    hist_calib = hist_calib / norm
    err_uncal = err_uncal / norm
    err_calib = err_calib / norm
    
    plt.errorbar(cbins,hist_uncal,yerr=err_uncal,color="black",fmt="o")
    plt.errorbar(cbins,hist_calib,yerr=err_calib,color="red",fmt="o")

    plt.xlabel("pixel value")
    plt.ylabel("rate per pixel per image")
    plt.yscale('log')
    plt.xlim(0,200)
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
    args = parser.parse_args()

    for filename in args.files:
        print "processing file:  ", filename
        process(filename, args)




