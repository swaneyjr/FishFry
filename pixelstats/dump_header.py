#! /usr/bin/env python

# dump the header from run data

import sys
import os
from unpack import *

import argparse

def process(filename,args):
    header = unpack_header(filename)
    show_header(header)

    if args.res:
        width = interpret_header(header,"width")
        height = interpret_header(header,"height")
        print("recording width:   ", width)
        print("recording height:  ", height)
        np.savez(os.path.join(args.calib, 'res.npz'), width=width, height=height)
    
if __name__ == "__main__":

    example_text = '''examples:

    ./dump_header.py data/FishStand/run_*_part_0_pixelstats.dat
    ./dump_header.py --res data/FishStand/run_*_part_0_pixelstats.dat'''
    
    parser = argparse.ArgumentParser(description='Combine multiple pixelstats data files.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--calib', default='calib', help="calibration directory to save resolution")
    parser.add_argument('--res', action='store_true', help='output resolution file')
    args = parser.parse_args()

    for filename in args.files:
        print "processing file:  ", filename
        process(filename, args)

        
