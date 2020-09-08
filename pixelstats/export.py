#!/usr/bin/env python3

import os
from unpack import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from hashlib import sha512

import argparse

HOT_NPZ = 'hot_online.npz'
HOT_CAL = 'hot_pixels.cal'

WGT_NPZ = 'lens.npz'
WGT_CAL = 'pixel_weight.cal'

DENOM = 1023

def export_hot(infile, outfile): 
 
    # load the hot cell list:
    try:
        hots  = np.load(infile)
    except:
        print("could not process file ", infile, " as .npz file.  Run gain.py with --commit option first?")
        return
    
    hot = hots['hot_list'].astype('>i4')
    hot_hash = int(sha512(hot.tostring()).hexdigest(), 16) & 0x7fffffff

    print("**** Hotcells ****")
    print("number of hot cells:         ", hot.size)
    print("maximum index:               ", np.max(hot))
    print("hash hot pixels:             ", hot_hash)
    print()
    
    #
    # Output to Java
    #
    
    h_dat = np.hstack([
        hot_hash, 
        hot.size, 
        hot]).astype('>i4')

    with open(outfile, 'w+') as f:
        h_dat.tofile(f)


def export_wgt(infile, outfile):

    # load gains
    try:
        gains  = np.load(infile);
    except:
        print("could not process file ", infile, " as .npz file.  Run gain.py with --commit option first?")
        return

    ds          = gains['down_sample']
    lens        = gains['lens']

    ly, lx = lens.shape

    wgt = np.where(lens > 0, 1/lens, 0)
    
    # normalize weights to [0,1]
    wgt /= wgt.max()
    wgt = wgt.astype('>f4')

    width  = lx*ds
    height = ly*ds
    wgt_hash = int(sha512(wgt.tostring()).hexdigest(), 16) & 0x7fffffff

    print("**** Weights ****")
    print("lens-shading map dims:   ", lens.shape)
    print("down sample:             ", ds)
    print("lx:                      ", lx)
    print("ly:                      ", ly)
    print("width:                   ", width)
    print("height:                  ", height)
    print("total pixels:            ", width*height) 

    print("max weight:              ", wgt.max())
    print("min weight:              ", wgt.min()) 
    print("mean weight:             ", wgt.mean())

    print("hash wgt:                ", wgt_hash) 
    print()
    
    #
    # Output to Java
    # 

    w_header = np.hstack([
        wgt_hash,
        ds,
        lx,
        ly]).astype(">i4")

    with open(outfile, 'w+') as f:
        w_header.astype(">i4").tofile(f)
        wgt.astype(">f4").tofile(f)
     

def import_hot(infile, outfile):

    try:
        f = open(infile)
    except:
        print(infile, 'not found')
        return

    hot_hash = np.fromfile(f, dtype='>i4',count=1)[0]
    n_hot = np.fromfile(f, dtype='>i4', count=1)[0]
    hot = np.fromfile(f,dtype='>i4')
    f.close()

    if hot.size != n_hot: 
        print('Invalid data in', infile)
        print('File length', hot.size, 'does not match metadata', n_hot)
        return

    if int(sha512(hot.tostring()).hexdigest(), 16) & 0x7fffffff != hot_hash:
        print('Invalid data in', infile)
        print('Computed hash value does not match metadata')
        return

    print("**** Hotcells ****")
    print("number of hot cells:         ", hot.size)
    print("maximum index:               ", np.max(hot))
    print("hash hot pixels:             ", hot_hash)
    print()
    
    #
    # Output to npz
    #
    
    np.savez(outfile, hot_list=hot)


def import_wgt(infile, outfile):
    
    # load .cal file
    try:
        f = open(infile)
    except:
        print(infile, 'not found')
        return

    wgt_hash = np.fromfile(f, dtype='>i4', count=1)[0]
    ds = np.fromfile(f, dtype='>i4', count=1)[0]
    lx = np.fromfile(f, dtype='>i4', count=1)[0]
    ly = np.fromfile(f, dtype='>i4', count=1)[0]
    wgt = np.fromfile(f, dtype='>f4')
    f.close()

    if wgt.size != lx*ly: 
        print('Invalid data in', infile)
        print('File length', wgt.size, 'does not match metadata', lx*ly)
        return
        
    if int(sha512(wgt.tostring()).hexdigest(), 16) & 0x7fffffff != wgt_hash:
        print('Invalid data in', infile)
        print('Computed hash value does not match metadata')
        return

    lens = np.where(wgt > 0, 1/wgt, 0).reshape(ly, lx)

    width  = lx*ds
    height = ly*ds
 
    print("**** Weights ****")
    print("lens-shading map dims:   ", (ly, lx))
    print("down sample:             ", ds)
    print("lx:                      ", lx)
    print("ly:                      ", ly)
    print("width:                   ", width)
    print("height:                  ", height)
    print("total pixels:            ", width*height) 

    print("max weight:              ", wgt.max())
    print("min weight:              ", wgt.min()) 
    print("mean weight:             ", wgt.mean())

    print("hash wgt:                ", wgt_hash) 
    print()
    
    #
    # Output to npz
    # 

    np.savez(outfile, down_sample=ds, lens=lens)

    
if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Export calibrations to Android', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--calib', default='calib', help='path to calibration files')
    parser.add_argument('--reverse', action='store_true', help='convert .cal files to .npz')
    args = parser.parse_args() 

    hot_npz = os.path.join(args.calib, HOT_NPZ)
    hot_cal = os.path.join(args.calib, HOT_CAL)
    wgt_npz = os.path.join(args.calib, WGT_NPZ)
    wgt_cal = os.path.join(args.calib, WGT_CAL)

    if args.reverse:
        import_hot(hot_cal, hot_npz)
        import_wgt(wgt_cal, wgt_npz)
    else:
        export_hot(hot_npz, hot_cal)
        export_wgt(wgt_npz, wgt_cal)


