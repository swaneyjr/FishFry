#!/usr/bin/env python3

# dump a header from run data

import sys
import argparse
from datetime import datetime
import numpy as np

from unpack_trigger import interpret_header, unpack_all
from calibrate import Calibrator

def log_framerate(filename):
    header,px,py,highest,region,timestamp,millistamp,images,dropped,millis_images = unpack_all(filename)

    tmin     = np.min(millis_images)
    tmax     = np.max(millis_images)
    elapsed  = (tmax - tmin)*1E-3
    exposure = interpret_header(header, "exposure")*1E-9
    duration = elapsed / images if images else 0
    dead     = (duration - exposure) / duration

    print("images:              ", images)
    print("first image:         ", tmin, " -> ", datetime.fromtimestamp(tmin*1E-3))   
    print("last image:          ", tmax, " -> ", datetime.fromtimestamp(tmax*1E-3))
    print("interval (s):        ", elapsed)
    print("frame duration (s):  ", duration)
    print("exposure (s):        ", exposure)
    print("deadtime frac:       ", dead)

def log_regions(filename, calibrator=None, triggered=True, zerobias=True):
    header,px,py,highest,region,timestamp,millistamp,images,dropped,millis_images = unpack_all(filename)

    if calibrator:
        region = calibrator.calibrate_region(px,py,region,header)

    dx = interpret_header(header,"region_dx")
    dy = interpret_header(header,"region_dy")

    num_region = region.shape[0]
    for i in range(num_region):
        h = highest[i]
        if not triggered and h==0: continue
        if not zerobias and h!=0: continue
        print("timestamp:    ", timestamp[i])
        print("millistamp:   ", millistamp[i], " -> ", datetime.fromtimestamp(millistamp[i]*1E-3))
        print("px:           ", px[i])
        print("py:           ", py[i])
        print("highest:      ", h)
        print("region:       ")
        print(region[i].reshape(2*dy+1,2*dx+1))
        print()

    print("images:                   ", images)
    print("total number of regions:  ", num_region)
    print("total dropped triggers:   ", dropped)
    print()
    
    
if __name__ == "__main__":

    example_text = ""
    
    parser = argparse.ArgumentParser(description='Combine multiple pixelstats data files.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--triggered',action="store_true", help="dump only triggered regions")
    parser.add_argument('--zerobias',action="store_true", help="dump only zero-bias regions")
    parser.add_argument('--calib',default='calib', help='location of calibration directory')
    parser.add_argument('--raw',action="store_true", help="do not apply weights to data")
    parser.add_argument('--framerate',action="store_true", help="calculate framerate and exit")

    args = parser.parse_args()

    if args.framerate:
        for filename in args.files:
            print('processing file:', filename)
            log_framerate(filename)
        quit()

    calibrator = Calibrator(args.calib) if not args.raw else None

    for filename in args.files:
        print("processing file:  ", filename)
        log_regions(filename, calibrator, 
                triggered=(not args.zerobias),
                zerobias=(not args.triggered))

        
