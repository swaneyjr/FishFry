#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt

import argparse

def process(filename, args):
    try:
        npz = np.load(filename)
    except:
        print "could not process file ", filename, " as .npz file."
        return
    
    for x in npz.iterkeys():
        print x
    
    a = npz['chan_a']
    b = npz['chan_b']
    c = npz['chan_c']

    ab  = np.intersect1d(a,b)
    bc  = np.intersect1d(b,c)
    ca  = np.intersect1d(c,a)
    abc = np.intersect1d(ab,c)

    print "a:   ", a.size
    print "b:   ", b.size
    print "c:   ", c.size
    print "ab:   ", ab.size
    print "bc:   ", bc.size
    print "ca:   ", ca.size
    print "abc:  ", abc.size

    print "efficiency is:  ", float(abc.size) / float(ab.size)

    print "min time:  ", np.min(a)
    print "max time:  ", np.max(a)



    
if __name__ == "__main__":
    example_text = '''examples:

    ./calibrate_time.py raw.npz'''
    
    parser = argparse.ArgumentParser(description='Calibrate Arduino times from heartbeat data.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--sandbox',action="store_true", help="run sandbox code and exit (for development).")
    parser.add_argument('--out',help="output filename")
    args = parser.parse_args()

    for filename in args.files:
        print "processing file:  ", filename
        process(filename, args)




