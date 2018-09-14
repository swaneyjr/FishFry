#! /usr/bin/env python

# dump a header from run data

import sys
import argparse

def process(filename,args):
    if (args.hist):
        import unpack_hist as unpack
    else:
        import unpack_trigger as unpack

    header = unpack.unpack_header(filename)
    unpack.show_header(header)

    if (args.geometry):
        width = unpack.interpret_header(header,"width")
        height = unpack.interpret_header(header,"height")
        print "recording width:   ", width
        print "recording height:  ", height
        np.savez("calib/geometry.npz", width=width, height=height)
    
if __name__ == "__main__":

    example_text = '''examples:

    ./dump_header.py data/FishStand/run_*_part_0_pixelstats.dat
    ./dump_header.py --geometry data/FishStand/run_*_part_0_pixelstats.dat'''
    
    parser = argparse.ArgumentParser(description='Combine multiple pixelstats data files.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--geometry',action="store_true", help="save geometry data to calibration file")
    parser.add_argument('--hist',action="store_true", help="dump header from a cosmic histogram file")
    args = parser.parse_args()

    if (args.geometry):
        if (len(args.files) != 1):
            print "specify only one file for defining the image geometry\n"
            exit(0)

    for filename in args.files:
        print "processing file:  ", filename
        process(filename, args)

        
