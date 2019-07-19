#!/usr/bin/env python3

import ROOT as r
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='')
    parser.add_argument('pfile', metavar='PFILE', help='ROOT file generated by mark_triggered.py')
    args = parser.parse_args()

    pfile = r.TFile(args.pfile)
    ptree = pfile.Get('triggers')
    
    bnames = [b.GetName() for b in ptree.GetListOfBranches()]
    if not 'triggered' in bnames:
        print("ERROR: use mark_triggered.py first")
        exit(1)

    triggered = []
    total = []

    tmin = 1e15
    tmax = 0

    for evt in ptree:
        tmin = min(tmin, evt.t)
        tmax = max(tmax, evt.t)
        if evt.triggered:
            triggered.append(max(evt.val))
        total.append(max(evt.val))

    dt = tmax - tmin

    print("Triggered: {} / {}".format(len(triggered), len(total)))

    bins = 2 * np.arange(15) ** 2 + 26
    #bins = range(27, 30) + range(30, 60, 10) + range(60, 160, 20) + range(160, 400, 40) 
    trig_hist, bins = np.histogram(triggered, bins)
    tot_hist, bins = np.histogram(total, bins)

    uinfo = ptree.GetUserInfo()
    rate = rate_err = tolerance = None
    for param in uinfo:
        if param.GetName() == 'tolerance':
            tolerance = param.GetVal()
        if param.GetName() == 'hodo_rate':
            rate = param.GetVal()
        if param.GetName() == 'hodo_rate_err':
            rate_err = param.GetVal()

    if not rate or not rate_err or not tolerance:
        print("ERROR: Metadata not found")
        exit(1)

    try:
        f_geom = np.load('calib/geometry.npz')
        
        # P(hodoscope | phone)
        p_hgp = f_geom['p_hgp']
        p_hgp_err = f_geom['p_hgp_err']
        
        # P(phone | hodoscope)
        p_pgh = f_geom['p_pgh']
        p_pgh_err = f_geom['p_pgh_err']

        # in Hz
        expected_rate = f_geom['rate']

        f_geom.close()
    except:
        print('ERROR: Geometry configuration not found. Try running "test_acceptance.py" first')
        exit(1)

    p_noise = 2 * tolerance * rate
    p_noise_err = 2 * tolerance * rate_err

    frac = (1.0 * trig_hist / tot_hist - p_noise) / (1 - p_noise) / p_hgp
    frac_err = frac * np.sqrt(\
            (p_hgp_err/p_hgp)**2 +  \
            (p_noise_err / (1 - p_noise))**2 + \
            (1./trig_hist + 1./tot_hist) / (1 - p_noise*tot_hist/trig_hist)**2)
    
    eff = (frac*tot_hist).sum() / (rate * dt * p_pgh / p_hgp)
    eff_err = eff * np.sqrt(\
            (np.sqrt(np.sum((frac_err*tot_hist)**2 + tot_hist*frac**2)) / (frac*tot_hist).sum())**2 + \
            (rate_err / rate)**2 + \
            (p_pgh_err / p_pgh)**2 + \
            (p_hgp_err / p_hgp)**2)


    print("observed rate: {0:.3f} mHz".format(rate * 1e3))
    print("expected rate: {0:.3f} mHz".format(expected_rate * 1e3))

    print("eff = {0:.4f} +\- {1:.4f}".format(eff, eff_err))

    plt.errorbar(bins[1:], frac, yerr=frac_err, fmt='o')
    plt.semilogx()
    plt.xlabel('Calibrated ADC counts')
    plt.ylabel('signal / total')
    plt.title('Signal fraction by max values')

    plt.show()
