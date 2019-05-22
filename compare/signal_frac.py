#!/usr/bin/env python

import ROOT as r
import numpy as np
import matplotlib.pyplot as plt

geom = 0.63

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='')
    parser.add_argument('pfile', metavar='PFILE', help='ROOT file generated by mark_triggered.py')
    args = parser.parse_args()

    pfile = r.TFile(args.pfile)
    ptree = pfile.Get('triggers')
    
    bnames = [b.GetName() for b in ptree.GetListOfBranches()]
    if not 'triggered' in bnames:
        print "ERROR: use mark_triggered.py first"
        print "Exiting"
        quit()

    triggered = []
    total = []

    for evt in ptree:
        if evt.triggered:
            triggered.append(max(evt.val))
        total.append(max(evt.val))

    print "Triggered: %d / %d" % (len(triggered), len(total))

    bins = 2 * np.arange(15) ** 2 + 26
    #bins = range(27, 30) + range(30, 60, 10) + range(60, 160, 20) + range(160, 400, 40) 
    trig_hist, bins = np.histogram(triggered, bins)
    tot_hist, bins = np.histogram(total, bins)

    uinfo = ptree.GetUserInfo()
    rate = tolerance = -1
    for param in uinfo:
        if param.GetName() == 'tolerance':
            tolerance = param.GetVal()
        if param.GetName() == 'hodo_rate':
            rate = param.GetVal()

    if rate == -1 or tolerance == -1:
        print "ERROR: Metadata not found"

    prob_noise = 2 * tolerance * rate

    frac = (1.0 * trig_hist / tot_hist - prob_noise) / (1 - prob_noise) / geom
    
    plt.errorbar(bins[1:], frac, yerr=frac*np.sqrt(1./trig_hist + 1./tot_hist), fmt='o')
    plt.semilogx()
    plt.xlabel('Calibrated ADC counts')
    plt.ylabel('signal / total')
    plt.title('Signal fraction by max values')

    plt.show()