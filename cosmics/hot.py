#!/usr/bin/env python3

# dump a header from run data

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from unpack_trigger import unpack_all, show_header, interpret_header
from calibrate import Calibrator 


def process(filename, calibrator, thresh=None, verbose=False):
    header,px,py,highest,region,timestamp,millistamp,images,dropped = unpack_all(filename)
    if verbose:
        show_header(header)

    region = calibrator.calibrate_region(px,py,region,header)
    icenter = region.shape[1] // 2

    # if thresh is set, use it to count occupancies
    # otherwise, use the the highese prescale
    idx_regions = region[:, icenter] > thresh if thresh \
            else (highest == highest.max())

    # return flattened indices with hits
    return py[idx_regions]*calibrator.width + px[idx_regions]
 
    
if __name__ == "__main__":

    example_text = ""
    
    parser = argparse.ArgumentParser(description='Offline (second pass) hot pixel finding.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--sandbox',action="store_true", help="run trial code")
    parser.add_argument('--thresh',  type=int, default=1, help="calibrated threshold for occupancy count")
    parser.add_argument('--calib', default='calib',help='location of calibration directory')
    parser.add_argument('--maxocc', default=1, type=int, help='maximum number of hits for "clean" pixel')
    parser.add_argument('--plot', action='store_true', help='plot pixel occupancies')
    parser.add_argument('--commit',action="store_true", help="save hot pixels to file.")
    parser.add_argument('--verbose', action="store_true", help="display file summary")

    args = parser.parse_args()

    calibrator = Calibrator(args.calib)
    total_pixels = calibrator.width * calibrator.height
    print('total pixels:', total_pixels)
    occ = np.zeros(total_pixels)

    end = "\n" if args.verbose else "\r"
    for filename in args.files:
        print("processing file:  ", filename, end=end)
        idx_occ = process(filename, 
                calibrator, 
                args.thresh, 
                verbose=False)
        occ += np.histogram(idx_occ, bins=np.arange(total_pixels+1))[0]
       
    print("max occupancy:  ", np.max(occ))
    print("total hits:     ", np.sum(occ))
    print("single hits:    ", np.sum(occ == 1))
    print("hot pixels:     ", np.sum(occ > args.maxocc))

    
    hot = np.argwhere(occ > args.maxocc).flatten()
    
    if args.plot:
        if occ.max() > 200:
            plt.hist(occ, bins=200, log=True)
        else:
            plt.hist(occ, bins=np.arange(occ.max()+1), log=True)
        plt.title('Pixel occupancy')
        plt.show()

    if args.commit:
        print("saving ", hot.size, " hot pixels to file.")
        np.savez(os.path.join(args.calib, 'hot_offline.npz'), hot_list=hot)

