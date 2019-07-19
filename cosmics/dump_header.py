#!/usr/bin/env python3

# dump a header from run data

import sys
import argparse
import numpy as np

def process(filename,args):
    if (args.hist):
        import unpack_hist as unpack
    else:
        import unpack_trigger as unpack

    header = unpack.unpack_header(filename)
    unpack.show_header(header)
    
if __name__ == "__main__":

    example_text = '''examples:

    ./dump_header.py data/FishStand/run_*_part_0_pixelstats.dat
    '''
    
    parser = argparse.ArgumentParser(description='Combine multiple pixelstats data files.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--hist',action="store_true", help="dump header from a cosmic histogram file")
    args = parser.parse_args()

    for filename in args.files:
        print("processing file:  ", filename)
        process(filename, args)

        
