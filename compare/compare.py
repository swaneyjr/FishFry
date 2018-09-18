#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt

import argparse
import datetime

def process(hfile, pfile, args):
    try:
        hnpz = np.load(hfile)
    except:
        print "could not process file ", hfile, " as .npz file."
        return

    try:
        pnpz = np.load(pfile)
    except:
        print "could not process file ", pfile, " as .npz file."
        return

    a = hnpz['chan_a']
    b = hnpz['chan_b']
    c = hnpz['chan_c']

    ab  = np.intersect1d(a,b)
    print "ab:   ", ab.size

    print ab

    p = pnpz['time']
    print p
    
    print "phone candidates:  ", p.size
    print "hodoscope muons:   ", ab.size

    print "phone times start at ", datetime.datetime.fromtimestamp(np.min(p))
    print "hodoscope times start at ", datetime.datetime.fromtimestamp(np.min(ab))
    print "phone times end at ", datetime.datetime.fromtimestamp(np.max(p))
    print "hodoscope times end at ", datetime.datetime.fromtimestamp(np.max(ab))


    imin = np.array([(np.abs(ab-p[i])).argmin() for i in range(p.size)])
    delta = p - ab[imin]


    h,bins = np.histogram(delta, bins=121, range=(-60,60))
    err = h**0.5
    cbins = 0.5*(bins[:-1] + bins[1:])
    plt.errorbar(cbins,h,yerr=err,color="black",fmt="o")
    plt.xlabel("delta t")
    plt.show()


    




    
if __name__ == "__main__":
    example_text = '''examples:

    ./calibrate_time.py raw.npz'''
    
    parser = argparse.ArgumentParser(description='Compre timestamps between phone and hodoscope.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('hfile', metavar='HFILE', help='hodoscope npz file')
    parser.add_argument('pfile', metavar='PFILE', help='phone npz file')
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    args = parser.parse_args()

    process(args.hfile, args.pfile, args)




