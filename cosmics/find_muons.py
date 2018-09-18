#!/usr/bin/env python

import sys
import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt

from unpack_trigger import *
from calibrate import *

all_time = np.array([])
all_trig = np.array([])


def process(filename,args):
    global all_time, all_trig, first_time
    header,px,py,highest,region,timestamp,millistamp,images,dropped = unpack_all(filename)

    try:
        filename = "calib/hot.npz"
        hots  = np.load(filename);
    except:
        print "could not process file ", filename, " as .npz file."
        return        
    hot_list = hots['hot']

    width  = interpret_header(header, "width")
    height = interpret_header(header, "height")

    dx = interpret_header(header,"region_dx")
    dy = interpret_header(header,"region_dy")
    region = calibrate_region(px,py,region,dx,dy)
    icenter = (2*dx + 1)*(2*dy + 1)/2
   
    threshold,prescale = get_trigger(header)

    index = py*width + px
    hot = np.in1d(index, hot_list)

    keep = (highest==prescale.size)
    print "found ", np.sum(hot[keep]), " hot regions."
    print "found ", np.sum(hot[keep] == False), " non-hot regions."


    keep = ((highest==prescale.size) & (hot == False))
    
    all_time = np.append(all_time, millistamp[keep]*1E-3)
    all_trig = np.append(all_trig,region[keep,icenter])
    

def analysis(args):
    global all_trig, all_time, first_time
    h,bins = np.histogram(np.clip(all_trig,0,500), bins=500, range=(0,500))
    cbins = 0.5*(bins[:-1] + bins[1:])
    plt.plot(cbins,h,"bo")
    plt.yscale('log')
    plt.show()

   
    time = np.unique(all_time[all_trig > 100])
    rate = time.size/time[-1]

    print "rate:  ", rate

    print "minimum time:  ", np.min(time)
    print "maximum time:  ", np.max(time)
    print "start date:    ", datetime.datetime.fromtimestamp(np.min(time))
    print "end date:      ", datetime.datetime.fromtimestamp(np.max(time))


    np.savez("muon.npz", time=time)



if __name__ == "__main__":
    example_text = '''examples:

    ...'''
    
    parser = argparse.ArgumentParser(description='Select muon candidates from cosmic data.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('trig', metavar='TRIG', nargs='+', help='trigger file(s) to process')
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    #parser.add_argument('--calib',action="store_true", help="compare calibrated pixel values.")
    parser.add_argument('--max',  type=int, default=50,help="maximum pixel value in rate plot (x-axis).")
    args = parser.parse_args()

    for filename in args.trig:
        print "processing trigger file:  ", filename
        process(filename, args)

    analysis(args)
