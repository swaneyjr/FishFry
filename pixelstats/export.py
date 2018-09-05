#!/usr/bin/env python

import sys
from unpack import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import argparse

from geometry import *

def analysis(args):

    # load the image geometry:
    try:
        geom = np.load("calib/geometry.npz");
    except:
        print "calib/geometry.npz does not exist.  Use dump_header.py --geometry"
        return
    width  = geom["width"]
    height = geom["height"]

    # load the lens shading map:
    try:
        filename = "calib/lens.npz"
        gains  = np.load(filename);
    except:
        print "could not process file ", filename, " as .npz file.  Run gain.py with --commit option first?"
        return
    ds          = gains['down_sample']
    lens        = gains['lens']
    

    # load the hot cell list:
    try:
        filename = "calib/hot.npz"
        hots  = np.load(filename);
    except:
        print "could not process file ", filename, " as .npz file.  Run gain.py with --commit option first?"
        return
    hot       = hots['hot_list']

    print "number of hot cells:        ", hot.size
    print "shape of lens shading map:  ", lens.shape
    print "down sample:                ", ds
    print "width:                      ", width
    print "height:                     ", height
    ly = lens.shape[0]
    lx = lens.shape[1]
    print "lx:                         ", lx
    print "ly:                         ", ly
    print "lx*ds:                      ", lx*ds
    print "ly*ds:                      ", ly*ds

    wgt = lens.reshape(lens.size)
    pos = (wgt > 0)
    wgt[pos] = 1.0/wgt[pos]
    wgt[pos==False] = 0

    print "zero weight regions:        ", np.sum(pos==False)
    print "weight 0:                   ", wgt[0]
    print "weight 1:                   ", wgt[1]

    h = np.array([],dtype=">i4")
    h = np.append(h,width)
    h = np.append(h,height)
    h = np.append(h,ds)
    h = np.append(h,lx)
    h = np.append(h,ly)

    with open('calib/pixel_weight.cal', 'w+') as f:
        h.astype(">i4").tofile(f)
        wgt.astype(">f4").tofile(f)

    h = np.array([])
    h = np.append(h,hot.size)
    h = np.append(h,hot)

    with open('calib/hot_pixels.cal', 'w+') as f:
        h.astype(">i4").tofile(f)



    return
    
    
if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Export calibrations to Android', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    args = parser.parse_args()

    analysis(args)




