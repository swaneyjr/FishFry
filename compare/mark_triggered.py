#!/usr/bin/env python3

import numpy as np
import ROOT as r

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

def find_cut(t0, trig_times, thresh=0):  
    alldiffs = np.array([evt.t_adj - trig_times for evt in t0 if max(evt.cal) > thresh])
    plt.hist(alldiffs.flatten(), bins=200, range=(-1000, 1000))
    plt.xlabel(r'$\Delta t$')
    plt.show()
    
    tmin = float(input("tmin: "))
    tmax = float(input("tmax: "))

    return tmin, tmax

def add_triggered(t0, hodo_times, trange=(0,0)): 

    tmin, tmax = trange

    t1 = t0.CloneTree(0)

    dt = np.sort(np.diff(np.sort(hodo_times))) 

    user_info = t1.GetUserInfo()
    user_info.Add(r.TParameter('Double_t')('tolerance', tmax-tmin))
    user_info.Add(r.TParameter('Double_t')('n_hodo', hodo_times.size))
    user_info.Add(r.TParameter('Double_t')('t_tot', dt.sum()))

    tag = np.zeros(1, dtype=bool)
    t1.Branch('tag', tag, 'tag/O') 

    n_tags = 0

    for i,evt in enumerate(t0):
        print(i, end='\r')
        diffs = evt.t_adj - hodo_times
        tag[0] = np.any((diffs > tmin) & (diffs < tmax))
        n_tags += tag[0]
        t1.Fill()

    print("Finished!")
    print("{} / {} triggers found".format(n_tags, t0.GetEntries()))

    return t1

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Add "triggered" branch to ROOT tree based on hodoscope data')
    parser.add_argument('--pfile', required=True, help='phone ROOT file')
    parser.add_argument('--hfile', required=True, help='hodoscope .npz file')
    parser.add_argument('--dt', type=float, default=600, help='Gap (in seconds) above which the hodoscope is considered to be turned off')
    parser.add_argument('--trange', type=int, nargs=2, help='Min and max time difference to trigger')
    parser.add_argument('--thresh', type=int, default=0, help='threshold for calibrated values') 
    parser.add_argument('--out', default='triggered.root', help='Output ROOT file')
    parser.add_argument('-a', action='store_true', help='Use LYSO A for coincidences')
    parser.add_argument('-b', action='store_true', help='Use LYSO B for coincidences')
    parser.add_argument('-c', action='store_true', help='Use LYSO C for coincidences')

    args = parser.parse_args()

    pfile = r.TFile(args.pfile)
    t0 = pfile.Get('triggers')

    pbranches = [b.GetName() for b in t0.GetListOfBranches()]
    if not 't_adj' in pbranches:
        print("ERROR: time corrections not yet set.")
        print("First, use correct_timestamps.py")
        exit()

    hodo = np.load(args.hfile)
    a = hodo['millis_a']
    b = hodo['millis_b']
    c = hodo['millis_c']

    dt = (-1,0,1)
  
    if args.a and args.b and not args.c:
        th = np.unique(np.hstack([np.intersect1d(a,b+t) for t in dt]))
    elif args.b and args.c and not args.a:
        th = np.unique(np.hstack([np.intersect1d(b,c+t) for t in dt]))
    elif args.c and args.a and not args.b:
        th = np.unique(np.hstack([np.intersect1d(c,a+t) for t in dt]))
    else:
        raise ValueError('Exactly two of -abc must be passed')

    outfile = r.TFile(args.out, 'recreate')

    trange = args.trange if args.trange else find_cut(t0, th)
    t1 = add_triggered(t0, th, trange) 

    outfile.Write()
    print("Wrote to", args.out)
    outfile.Close()
    pfile.Close()
    
