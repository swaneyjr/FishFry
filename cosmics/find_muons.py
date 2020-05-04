#!/usr/bin/env python3

import sys
import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from unpack_trigger import unpack_all, get_trigger
from calibrate import Calibrator

import ROOT as r


def process(filename, calibrator, hot_list=None, verbose=False):

    header,px,py,highest,raw_region,timestamp,millistamp,images,dropped = unpack_all(filename)

    cal_region = calibrator.calibrate_region(px,py,raw_region,header)
    icenter = raw_region.shape[1]//2
   
    threshold,prescale = get_trigger(header)
 
    if np.any(hot_list):
        index = py*calibrator.width + px
        hot = np.in1d(index, hot_list)
        keep = (highest==prescale.size) & np.logical_not(hot)
    
    else:
        keep = (highest==prescale.size)
    
    if verbose:
        print("found ", np.sum(hot), " hot regions.")
        print("found ", np.sum(np.logical_not(hot)), " non-hot regions.") 
    
    t = millistamp[keep]*1E-3
    x = px[keep]
    y = py[keep]
    raw = raw_region[keep,icenter]
    cal = cal_region[keep,icenter]

    return t, x, y, raw, cal
 

def save_trig(fname, t, x, y, raw, cal):
    f = r.TFile(args.out, 'recreate')
    trigs = r.TTree('triggers', 'Triggered events')

    t0 = np.array([t.min()])
    vx = r.vector('UInt_t')()
    vy = r.vector('UInt_t')()
    vraw = r.vector('UInt_t')()
    vcal = r.vector('UInt_t')()

    trigs.Branch('t', vt, 't/D')
    trigs.Branch('x', vx)
    trigs.Branch('y', vy)
    trigs.Branch('raw', vraw)
    trigs.Branch('cal', vcal)

    for i in np.argsort(t):
        if t[i] > t0:
            trigs.Fill()
            vx.clear()
            vy.clear()
            vraw.clear()
            vcal.clear()
            t0[0] = t[i]
        
        vx.push_back(int(x[i]))
        vy.push_back(int(y[i]))
        vraw.push_back(int(raw[i]))
        vcal.push_back(int(trig[i]))

    trigs.Fill()

    f.Write()
    f.Close()


if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Select muon candidates from cosmic data.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('trig', metavar='TRIG', nargs='+', help='trigger file(s) to process')
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--calib', default='calib', help="compare calibrated pixel values.")
    parser.add_argument('--max',  type=int, default=50,help="maximum pixel value in rate plot (x-axis).")
    parser.add_argument('--plot', action='store_true',help='plot histogram of triggered values')
    parser.add_argument('--out', help='name of output ROOT file')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose output')
    args = parser.parse_args() 

    end = "\n" if args.verbose else "\r"

    hot_list = None
    hot_fname = os.path.join(args.calib, "hot_offline.npz")

    try:
        hotf  = np.load(hot_fname)
        hot_list = hotf['hot_list']
        hotf.close()
    except:
        print("could not process file ", hot_fname, " as .npz file.") 

    calibrator = Calibrator(args.calib)

    # update these lists
    all_t   = []  
    all_x   = []
    all_y   = []
    all_raw = []
    all_cal = []

    for filename in args.trig:
        print("processing trigger file:  ", filename, end=end)
        t, x, y, raw, cal = process(filename, calibrator, hot_list, args.verbose)
        
        all_t   += list(t)
        all_x   += list(x)
        all_y   += list(y)
        all_raw += list(raw)
        all_cal += list(cal)

    all_t   = np.array(all_t)
    all_x   = np.array(all_x)
    all_y   = np.array(all_y)
    all_raw = np.array(all_raw)
    all_cal = np.array(all_cal)

    time = np.unique(all_t)
    rate = time.size / (time.max() - time.min())
    
    print()

    print("rate:            ", rate)
    print("minimum time:    ", np.min(time))
    print("maximum time:    ", np.max(time))
    print("start date:      ", datetime.fromtimestamp(np.min(time)))
    print("end date:        ", datetime.fromtimestamp(np.max(time)))

    if args.plot:
        plt.hist(all_cal, bins=np.arange(500), log=True)
        plt.show()

    if args.out:
        save_trig(args.out, all_t, all_x, all_y, all_raw, all_cal)
