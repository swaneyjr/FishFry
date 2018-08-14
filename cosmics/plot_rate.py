#!/usr/bin/env python

import sys
from unpack import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import argparse

def process(filename, args):

    # load data:
    header,hist = unpack_all(filename)
    images = interpret_header(header, "images")
    width  = interpret_header(header, "width")
    height = interpret_header(header, "height")
    
    print hist.size
    print hist

    cbins = np.arange(0,hist.size) + 0.5
    err = hist**0.5
    plt.errorbar(cbins,hist,yerr=err,color="black",fmt="o")
    #plt.ylabel("pixels")
    #plt.xlabel("mean")
    #plt.yscale('log')
    #plt.savefig("hmean.pdf")
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




