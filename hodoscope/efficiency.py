#!/usr/bin/env python3

import numpy as np

import argparse

def process(filename, center):
    try:
        npz = np.load(filename)
    except:
        print("could not process file ", filename, " as .npz file.")
        return
    
    #for x in npz.iterkeys():
    #    print(x)
    
    a = npz['chan_a']
    b = npz['chan_b']
    c = npz['chan_c']

    ab  = np.intersect1d(a,b)
    bc  = np.intersect1d(b,c)
    ca  = np.intersect1d(c,a)
    abc = np.intersect1d(ab,c)

    print("a:   ", a.size)
    print("b:   ", b.size)
    print("c:   ", c.size)
    print("ab:  ", ab.size)
    print("bc:  ", bc.size)
    print("ca:  ", ca.size)
    print("abc: ", abc.size)
    print()

    if center == 'a':
        denom = bc.size
    elif center == 'b':
        denom = ac.size
    elif center == 'c':
        denom = ab.size

    eff = abc.size / denom
    eff_err = np.sqrt(eff * (1-eff) / denom)

    print("eff = ", eff, "+/-", eff_err)

    print("min time:  ", np.min(a))
    print("max time:  ", np.max(a))

    
if __name__ == "__main__":
    example_text = '''examples:

    ./calibrate_time.py raw.npz'''
    
    parser = argparse.ArgumentParser(description='Calibrate Arduino times from heartbeat data.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--out',help="output filename")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-a', action='store_true')
    group.add_argument('-b', action='store_true')
    group.add_argument('-c', action='store_true')
    args = parser.parse_args()

    if args.a:
        center = 'a'
    elif args.b:
        center = 'b'
    elif args.c:
        center = 'c'

    for filename in args.files:
        print("processing file:  ", filename)
        process(filename, center)




