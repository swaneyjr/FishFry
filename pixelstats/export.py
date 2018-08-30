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
    ny = lens.shape[0]
    nx = lens.shape[1]
    print "nx:                         ", nx
    print "ny:                         ", ny
    print "nx*ds:                      ", nx*ds
    print "ny*ds:                      ", ny*ds

    wgt = lens.reshape(lens.size)
    pos = (wgt > 0)
    wgt[pos] = 1.0/wgt[pos]
    wgt[pos==False] = 0

    print "zero weight regions:        ", np.sum(pos==False)
    print "weight 0:                   ", wgt[0]
    print "weight 1:                   ", wgt[1]

    h = np.array([],dtype=">i4")
    h = np.append(h,ds)
    h = np.append(h,nx)
    h = np.append(h,ny)

    p = np.array([],dtype=">f4")
    p = np.append(p,wgt)

    with open('calib/weight.dat', 'w+') as f:
        h.tofile(f)
        p.tofile(f)

    h = np.array([],dtype=">i4")
    h = np.append(h,hot.size)
    h = np.append(h,hot)

    with open('calib/hot.dat', 'w+') as f:
        h.tofile(f)



    return
    
    
if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Export calibrations to Android', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    args = parser.parse_args()

    analysis(args)




