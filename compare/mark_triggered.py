#!/usr/bin/env python

import numpy as np
import ROOT as r

def find_cut(t0, trig_times, cutoff=50):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('text', usetex=True)

    alldiffs = np.array([evt.t - trig_times for evt in t0 if max(evt.val) > cutoff])
    plt.hist(alldiffs.flatten(), bins=60, range=(-10, 10))
    plt.xlabel(r'$\Delta t$')
    plt.show()
    
    offset = float(raw_input("Offset: "))
    tolerance = float(raw_input("Tolerance: "))

    return offset, tolerance

def add_triggered(t0, trig_times, offset=0, tolerance=1): 

    t1 = t0.CloneTree(0)

    hodo_rate = 1.0*trig_times.size / (trig_times.max() - trig_times.min())

    user_info = t1.GetUserInfo()
    user_info.Add(r.TParameter('Double_t')('offset', offset))
    user_info.Add(r.TParameter('Double_t')('tolerance', tolerance))
    user_info.Add(r.TParameter('Double_t')('hodo_rate', hodo_rate))

    triggered = np.zeros(1, dtype=bool)
    t1.Branch('triggered', triggered, 'triggered/O') 

    trigs = 0

    for evt in t0:
        triggered[0] = np.amin(np.abs(evt.t - trig_times - offset)) < tolerance
        trigs += triggered[0]
        t1.Fill()

    #noise = trig_times.size / (trig_times.max() - trig_times.min()) * 1.5 
    #user_info.Add(r.TParameter('Double_t')('noise', noise))

    print "Finished!"
    print "%d / %d triggers found" % (trigs, t0.GetEntries())

    return t1

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Add "triggered" branch to ROOT tree based on hodoscope data')
    parser.add_argument('--pfile', required=True, help='phone ROOT file')
    parser.add_argument('--hfile', required=True, help='hodoscope .npz file')
    parser.add_argument('--out', default='triggered.root', help='Output ROOT file')

    args = parser.parse_args()

    pfile = r.TFile(args.pfile)
    t0 = pfile.Get('triggers')

    hfile = np.load(args.hfile)
    htimes = np.intersect1d(hfile.f.chan_a, hfile.f.chan_b)

    outfile = r.TFile(args.out, 'recreate')
    t1 = add_triggered(t0, htimes, *find_cut(t0, htimes)) 

    outfile.Write()
    print "Wrote to %s" % args.out
    outfile.Close()
    
