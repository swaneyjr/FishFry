#!/usr/bin/env python3

# dump a header from run data

import sys
import argparse
import numpy as np
from unpack_trigger import unpack_all, show_header, interpret_header
from calibrate import calibrate_region 

width = None
height = None
total_pixels = 0
occ = np.array([])


def process(filename,args):
    header,px,py,highest,region,timestamp,millistamp,images,dropped = unpack_all(filename)
    if args.verbose:
        show_header(header)

    global width, height, total_pixels, occ
    if not width or not height:
        print("initializing...")
        width = interpret_header(header,"width")
        height = interpret_header(header,"height")
        total_pixels = width*height
        print("total pixels:  ", total_pixels)
        occ = np.zeros(total_pixels)

    dx = interpret_header(header,"region_dx")
    dy = interpret_header(header,"region_dy")
    region = calibrate_region(px,py,region,dx,dy,width,height)
    icenter = ((2*dx + 1)*(2*dy+1)) // 2

    num_region = region.shape[0]
    count = 0
    for i in range(num_region):
        if not highest[i]:
            continue;
        if region[i][icenter] > args.thresh:
            index = py[i]*width + px[i]
            occ[index] += 1

def analysis(args):
    print("max occupancy:  ", np.max(occ))
    print("total hits:     ", np.sum(occ))
    print("single hits:    ", np.sum(occ == 1))
    print("hot pixels:     ", np.sum(occ > 1))

    index = np.arange(width*height)
    hot = index[occ > 1]

    if (args.commit):
        print("saving ", hot.size, " hot pixels to file.")
        np.savez("calib/hot.npz", hot=hot)

    
if __name__ == "__main__":

    example_text = ""
    
    parser = argparse.ArgumentParser(description='Offline (second pass) hot pixel finding.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--sandbox',action="store_true", help="run trial code")
    parser.add_argument('--thresh',  type=float, default=27,help="calibrated threshold for occupancy count")
    parser.add_argument('--commit',action="store_true", help="save hot pixels to file.")
    parser.add_argument('--verbose', action="store_true", help="display file summary")

    args = parser.parse_args()

    end = "\n" if args.verbose else "\r"
    for filename in args.files:
        print("processing file:  ", filename, end=end)
        process(filename, args)
       
    analysis(args)
