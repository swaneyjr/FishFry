#! /usr/bin/env python

# dump a header from run data

import sys
import argparse

from unpack_trigger import *
from calibrate import *

def process(filename,args):
    header,px,py,highest,region,images,dropped = unpack_all(filename)
    show_header(header)

    if (args.calib):
        dx = interpret_header(header,"region_dx")
        dy = interpret_header(header,"region_dy")
        region = calibrate_region(px,py,region,dx,dy)

    num_region = region.shape[0]
    count = 0
    for i in range(num_region):
        h = highest[i];
        if (args.triggered and (h==0)):
            continue
        if (args.zerobias and (h!=0)):
            continue
        print "px:       ", px[i]
        print "py:       ", py[i]
        print "highest:  ", h
        print "region:   ", region[i]
        count += 1
        if (args.short):
            if (count >= 100):
                break

    print "images:                   ", images
    print "total number of regions:  ", num_region
    print "regions shown:            ", count
    print "total dropped triggers:   ", dropped
    
    
if __name__ == "__main__":

    example_text = ""
    
    parser = argparse.ArgumentParser(description='Combine multiple pixelstats data files.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--triggered',action="store_true", help="dump only triggered regions")
    parser.add_argument('--zerobias',action="store_true", help="dump only zero-bias regions")
    parser.add_argument('--short',action="store_true", help="run over small amount of data")
    parser.add_argument('--calib',action="store_true", help="apply calibrated weights to region data")

    args = parser.parse_args()

    for filename in args.files:
        print "processing file:  ", filename
        process(filename, args)

        
