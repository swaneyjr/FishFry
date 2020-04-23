#!/usr/bin/env python3

import sys
import os
import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt

from unpack_trigger import *
from calibrate import *

import ROOT as r

all_time = np.array([])
all_trig = np.array([]).astype(int)
all_raw = np.array([]).astype(int)
all_x = np.array([]).astype(int)
all_y = np.array([]).astype(int)

width = None
height = None

def process(filename, calib, hot_list=None, verbose=False):

    global all_time, all_trig, all_raw, all_x, all_y
    global width, height
    header,px,py,highest,raw_region,timestamp,millistamp,images,dropped = unpack_all(filename)
 
    if not width or not height:
        width  = interpret_header(header, "width")
        height = interpret_header(header, "height")

    dx = interpret_header(header,"region_dx")
    dy = interpret_header(header,"region_dy")
    region = calibrate_region(px,py,raw_region,dx,dy,width,height,calib)
    icenter = (2*dx + 1)*(2*dy + 1)//2
   
    threshold,prescale = get_trigger(header)
 
    if np.any(hot_list):
        index = py*width + px
        hot = np.in1d(index, hot_list)
        keep = (highest==prescale.size) & np.logical_not(hot)
    
    else:
        keep = (highest==prescale.size)
    
    if verbose:
        print("found ", np.sum(hot), " hot regions.")
        print("found ", np.sum(np.logical_not(hot)), " non-hot regions.") 
    

    all_time = np.append(all_time, millistamp[keep]*1E-3)
    all_raw = np.append(all_trig,raw_region[keep,icenter])
    all_trig = np.append(all_trig,region[keep,icenter])
    all_x = np.append(all_x, px)
    all_y = np.append(all_y, py)


def analysis():
    global all_trig, all_time, all_x, all_y
    plt.hist(all_trig, bins=np.arange(500), log=True)
    plt.show()

    time = np.unique(all_time)
    rate = time.size/time[-1]

    
    print()

    print("rate:  ", rate)

    print("minimum time:  ", np.min(time))
    print("maximum time:  ", np.max(time))
    print("start date:    ", datetime.datetime.fromtimestamp(np.min(time)))
    print("end date:      ", datetime.datetime.fromtimestamp(np.max(time)))



if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Select muon candidates from cosmic data.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('trig', metavar='TRIG', nargs='+', help='trigger file(s) to process')
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--calib', default='calib', help="compare calibrated pixel values.")
    parser.add_argument('--max',  type=int, default=50,help="maximum pixel value in rate plot (x-axis).")
    parser.add_argument('--out', default='phone.root', help='name of output ROOT file')
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

    for filename in args.trig:
        print("processing trigger file:  ", filename, end=end)
        process(filename, args.calib, hot_list, args.verbose)

    analysis()

    f = r.TFile(args.out, 'recreate')
    trigs = r.TTree('triggers', 'Triggered events')

    times = np.array([all_time.min()])
    x = r.vector('UInt_t')()
    y = r.vector('UInt_t')()
    raw = r.vector('UInt_t')()
    val = r.vector('UInt_t')()

    trigs.Branch('t', times, 't/D')
    trigs.Branch('x', x)
    trigs.Branch('y', y)
    trigs.Branch('raw_val', raw)
    trigs.Branch('val', val)

    for i in np.argsort(all_time):
        if all_time[i] > times[0]:
            trigs.Fill()
            x.clear()
            y.clear()
            raw.clear()
            val.clear()
            times[0] = all_time[i]
        
        x.push_back(int(all_x[i]))
        y.push_back(int(all_y[i]))
        raw.push_back(int(all_raw[i]))
        val.push_back(int(all_trig[i]))

    trigs.Fill()

    f.Write()
    f.Close()
