#! /usr/bin/env python

# dump a header from run data

import sys
import argparse
import datetime

from unpack_trigger import *
from calibrate import *

def process(filename,args):
    header,px,py,highest,region,timestamp,millistamp,images,dropped = unpack_all(filename)

    if (args.framerate):
        tmin     = np.min(millistamp)
        tmax     = np.max(millistamp)
        elapsed  = (tmax - tmin)*1E-3
        exposure = interpret_header(header, "exposure")*1E-9
        if (images > 0):
            duration = elapsed / images
        else:
            duration = 0
        dead     = (duration - exposure) / duration

        print "images:              ", images
        print "first image:         ", tmin, " -> ", datetime.datetime.fromtimestamp(tmin*1E-3)    
        print "last image:          ", tmax, " -> ", datetime.datetime.fromtimestamp(tmax*1E-3)     
        print "interval (s):        ", elapsed  
        print "frame duration (s):  ", duration
        print "exposure (s):        ", exposure
        print "deadtime frac:       ", dead
        return




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
        print "timestamp:    ", timestamp[i]
        print "millistamp:   ", millistamp[i], " -> ", datetime.datetime.fromtimestamp(millistamp[i]*1E-3)
        print "px:           ", px[i]
        print "py:           ", py[i]
        print "highest:      ", h
        print "region:       ", region[i]
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
    parser.add_argument('--framerate',action="store_true", help="calculate framerate and exit")

    args = parser.parse_args()

    for filename in args.files:
        print "processing file:  ", filename
        process(filename, args)

        
