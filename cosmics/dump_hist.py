#!/usr/bin/env python3

import sys
from unpack_hist import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import argparse

def process(filename, args):

    # load data:
    header,hist_uncal,hist_nohot,hist_calib = unpack_all(filename)

    show_header(header)

    images        = interpret_header(header, "images")
    width         = interpret_header(header, "width")
    height        = interpret_header(header, "height")
    hist_prescale = interpret_header(header, "hist_prescale")

    
    print("images:         ", images)
    print("width:          ", width)
    print("height:         ", height)
    print("hist_prescale:  ", hist_prescale)

    print("max entries:    ", images * width * height / hist_prescale)
    print("buffer size:    ", 2**31)
    
    print("sum of entries, uncal:  ", np.sum(hist_uncal))
    print("sum of entries, unhot:  ", np.sum(hist_unhot))
    print("sum of entries, calib:  ", np.sum(hist_calib))

    print(hist_uncal[:100])
    print(hist_unhot[:100])
    print(hist_calib[:100])


if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Plot rate from Cosmics.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    args = parser.parse_args()

    for filename in args.files:
        print("processing file:  ", filename)
        process(filename, args)




