#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from unpack_hist import unpack_all, show_header, interpret_header

import argparse

def process(filename, args):

    # load data:
    header,hist_cln,hist_hot,hist_cal = unpack_all(filename)

    show_header(header)

    images        = interpret_header(header, "images")
    width         = interpret_header(header, "width")
    height        = interpret_header(header, "height")
    hist_prescale = interpret_header(header, "hist_prescale")

    # need to convert to python primitive types to prevent overflow
    print("max entries:    ", int(images) * int(width) * int(height) // hist_prescale)
    print("buffer size:    ", 2**63)
    
    print("total samples, clean:  ", np.sum(hist_cln))
    print("total samples, hot:    ", np.sum(hist_hot))
    print("total samples, calib:  ", np.sum(hist_cal))

    print('clean:', hist_cln[:100], '...\n')
    print('hot:  ', hist_hot[:100], '...\n')
    print('calib:', hist_cal[:100], '...')


if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Print histogram data.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    args = parser.parse_args()

    for filename in args.files:
        print("processing file:  ", filename)
        process(filename, args)




