#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

import ROOT as r

def find_cut(t0, hodo_times, thresh=0, noise=0, nbins=50, dtmax=1000, duration=None):  
    print('Computing diffs')
    alldiffs = []
    for i,evt in enumerate(t0):
        print('{} / {}'.format(i+1, t0.GetEntries()), end='\r')
        if evt.max >= thresh:
            for th in hodo_times:
                diffs = evt.t_adj - th
                alldiffs.append(diffs[np.abs(diffs) < dtmax])
    
    alldiffs = np.hstack(alldiffs)
    
    # plot histogram of dt
    hist, bins = np.histogram(alldiffs, bins=np.linspace(-dtmax, dtmax, nbins))
    bincenters = (bins[1:] + bins[:-1])/2
    
    plt.figure(figsize=(4,3), tight_layout=True)
    ax = plt.gca()
    ax.errorbar(bincenters, hist, yerr=np.sqrt(hist), ls='', marker='o', ms=3, color='k', label='Data')
    ax.plot(bins, noise*(bins[1]-bins[0])*np.ones(bins.size), 'r-', label='Background')
    if duration:
        ax.axvspan(-duration/2, duration/2, alpha=0.5, color='darkgray', label='Frame duration')

    #plt.xlabel(r'$t_\mathrm{phone} - t_\mathrm{hodo}$ (ms)')
    plt.xlabel(r'$\Delta t$ (CMOS, scintillator) [ms]')
    plt.ylabel('CMOS-scintillator pairs')
    plt.legend()
    plt.show()
   
    print()
    tmin = float(input("tmin: "))
    tmax = float(input("tmax: "))

    return tmin, tmax

def add_tags(t0, hodo_times, trange=(0,0)): 

    tmin, tmax = trange

    t1 = t0.CloneTree(0)

    tag = {c: np.zeros(1, dtype=bool) for c in hodo_times}
    n_tags = {c: 0 for c in hodo_times}

    for k,v in tag.items():
        t1.Branch('tag_{}'.format(k), v, 'tag/O')

    for i,evt in enumerate(t0):
        print(i, '/', t0.GetEntries(), end='\r')
        for c, th in hodo_times.items():
            diffs = evt.t_adj - th
            tag[c][0] = np.any((diffs > tmin) & (diffs < tmax))
            n_tags[c] += tag[c][0]
        t1.Fill()

    print("Finished!")
    print("{} / {} triggers found".format(n_tags, t0.GetEntries()))

    return t1


def get_blocks(name, trig, nontrig, hodo, trange): 

    tmin, tmax = trange

    t_blocks = r.TTree(name, 'Independent frame blocks')

    t_ = np.array([0], dtype=int)
    M_ = np.array([0], dtype=int)
    N_ = np.array([0], dtype=int)
    max_ = np.array([0], dtype=int)
    sum5_ = np.array([0], dtype=int)
    sum9_ = np.array([0], dtype=int)
    sum21_ = np.array([0], dtype=int)

    t_blocks.Branch('t', t_, 't/l')
    t_blocks.Branch('M', M_, 'M/i')
    t_blocks.Branch('N', N_, 'N/i')
    t_blocks.Branch('max', max_, 'max/i')
    t_blocks.Branch('sum5', sum5_, 'sum5/i')
    t_blocks.Branch('sum9', sum9_, 'sum9/i')
    t_blocks.Branch('sum21', sum21_, 'sum21/i')

    th = np.hstack([[-np.inf], hodo, [np.inf]])
    trig_times = [tr.t_adj for tr in trig] + [np.inf]
    nontrig_times = [n.t_adj for n in nontrig] + [np.inf]

    trig_idx = 0
    nontrig_idx = 0
    hodo_idx = 0

    trig.GetEntry(trig_idx)
    nontrig.GetEntry(trig_idx)

    trig_entries = len(trig_times) - 1
    nontrig_entries = len(nontrig_times) - 1
    hodo_entries = len(th) 

    # get first event, regardless of source
    triggered = trig_times[0] < nontrig_times[0]
    if triggered:
        evt = trig
        trig_idx += 1
    else:
        evt = nontrig
        nontrig_idx += 1

    trig_t_next = trig_times[trig_idx]
    nontrig_t_next = nontrig_times[nontrig_idx]

    t_[0] = evt.t_adj 

    while True: 

        # first decide whether to add to existing block 
        # or start a new one
        #print(evt.t_adj, triggered, evt.t_adj - th[hodo_idx])
        while evt.t_adj - th[hodo_idx+1] >= tmin:
            M_[0] += 1
            hodo_idx += 1

        # update current block
        N_[0] += 1
        if triggered:
            max_[0]   = max(max_[0],   evt.max)
            # quit here for mc
            if hasattr(evt, 'sum5'):
                sum5_[0]  = max(sum5_[0],  evt.sum5)
                sum9_[0]  = max(sum9_[0],  evt.sum9)
                sum21_[0] = max(sum21_[0], evt.sum21)

        # load next timestamp
        if trig_idx == trig_entries and nontrig_idx == nontrig_entries:
            # all timestamps have been iterated through
            t_blocks.Fill()
            return t_blocks

        # check whether to draw next event from triggered
        # or nontriggered datasets
        triggered = trig_t_next < nontrig_t_next

        if triggered:
            trig.GetEntry(trig_idx)
            trig_idx += 1
            evt = trig
            trig_t_next = trig_times[trig_idx]
        else:
            nontrig.GetEntry(nontrig_idx)
            nontrig_idx += 1
            evt = nontrig
            nontrig_t_next = nontrig_times[nontrig_idx]
        

        if evt.t_adj - th[hodo_idx] >= tmax:

            t_blocks.Fill()
            
            t_[0] = evt.t_adj
            M_[0] = 0 
            N_[0] = 0
            max_[0] = 0
            sum5_[0] = 0
            sum9_[0] = 0
            sum21_[0] = 0




if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Add "triggered" branch to ROOT tree based on hodoscope data')
    parser.add_argument('--pfile', required=True, nargs='+', help='phone ROOT file')
    parser.add_argument('--hfile', required=True, help='hodoscope .npz file')
    parser.add_argument('--dtmax', type=float, default=1000, help='Range in ms for histogram')
    parser.add_argument('--trange', type=int, nargs=2, help='Min and max time difference to trigger')
    parser.add_argument('--thresh', type=int, default=0, help='threshold for calibrated values') 
    parser.add_argument('--nbins', type=int, default=50, help='Number of bins to use for plotting')
    parser.add_argument('--duration', type=int, help='Frame duration to highlight in plot')
    parser.add_argument('--ab', action='store_true', help='Include AB coincidences in plot')
    parser.add_argument('--ac', action='store_true', help='Include AC coincidences in plot')
    parser.add_argument('--bc', action='store_true', help='Include BC coincidences in plot')
    parser.add_argument('--out', help='Output ROOT file')

    args = parser.parse_args()

    # load hodoscope data and compare time bounds
    hodo = np.load(args.hfile) 

    # load phone data
    t0_pre = r.TChain('triggers')
    tn_pre = r.TChain('nontriggers')

    print('Loading data')
    th_intervals = []
    t_tot = 0

    for pfile in args.pfile:
        print(pfile)

        pf = r.TFile(pfile)
        t0 = pf.Get('triggers')
        tn = pf.Get('nontriggers')

        pbranches = [b.GetName() for b in t0.GetListOfBranches()]
        if not 't_adj' in pbranches:
            print("ERROR: time corrections not yet set.")
            print("First, use correct_timestamps.py")
            exit()

        # find bounds in time of phone run
        tpmin = 1e15
        tpmax = 0
        
        for evt in t0:
            tpmin = min(tpmin, evt.t_adj)
            tpmax = max(tpmax, evt.t_adj)
        for evt in tn:
            tpmin = min(tpmin, evt.t_adj)
            tpmax = max(tpmax, evt.t_adj)

        for thi,thf in zip(hodo['interval_ti'], hodo['interval_tf']):
            thmin = max(tpmin, thi)
            thmax = min(tpmax, thf)
            
            # check if the phone and hodo intervals actually intersect
            if thmax - thmin <= 0: continue
        
            t_tot += thmax - thmin
            th_intervals.append((thmin, thmax))

        pf.Close()

        t0_pre.Add(pfile)
        tn_pre.Add(pfile)
 
    print('Copying TTrees')
    r.gROOT.cd()
    cut_str = ' || '.join(['t_adj > {} && t_adj < {}'.format(thi, thf) for thi,thf in th_intervals])
    t0 = t0_pre.CopyTree(cut_str)
    tn = tn_pre.CopyTree(cut_str)

    a = hodo['millis_a']
    b = hodo['millis_b']
    c = hodo['millis_c']

    dt = (-1,0,1)
 
    hodo_times = {
            'AB': np.unique(np.hstack([np.intersect1d(a,b+t) for t in dt])),
            'BC': np.unique(np.hstack([np.intersect1d(b,c+t) for t in dt])),
            'AC': np.unique(np.hstack([np.intersect1d(c,a+t) for t in dt])),
        }

    # turn this into a list for plotting
    hodo_times_plot = []
    if args.ab:
        hodo_times_plot.append(hodo_times['AB'])
    if args.ac:
        hodo_times_plot.append(hodo_times['AC'])
    if args.bc:
        hodo_times_plot.append(hodo_times['BC'])

    if not args.ab and not args.ac and not args.bc:
        hodo_times_plot = hodo_times.values()

    hodo_count = sum(len(l) for l in hodo_times_plot)
    hodo_time = np.sum(hodo['interval_tf'] - hodo['interval_ti'])
    phone_count = t0.GetEntries('max >= {}'.format(args.thresh))
    noise_rate = hodo_count / hodo_time * phone_count 
    
    hodo.close() 

    if not args.out:
        find_cut(t0, hodo_times_plot, 
                args.thresh, 
                dtmax = args.dtmax,
                nbins=args.nbins,
                noise=noise_rate,
                duration=args.duration)
        exit(0)

    trange = args.trange if args.trange else \
            find_cut(t0, hodo_times_plot, 
                    args.thresh,
                    nbins=args.nbins,
                    dtmax=args.dtmax, 
                    noise=noise_rate,
                    duration=args.duration) 

    # write file with new TTrees
    outfile = r.TFile(args.out, 'recreate')
    print('Writing!')

    t_blocks = []
    print('Building blocks')
    for c, th in hodo_times.items():
        print(c)

        tcuts = [(th >= thmin) & (th <= thmax) for thmin, thmax in th_intervals]
        th = th[np.logical_or.reduce(tcuts)]

        t_blocks_temp = get_blocks('blocks_{}'.format(c), t0, tn, th, trange)
      
        user_info = t_blocks_temp.GetUserInfo()
        user_info.Add(r.TParameter('Double_t')('tolerance', trange[1] - trange[0]))
        user_info.Add(r.TParameter('Double_t')('n_hodo', th.size))
        user_info.Add(r.TParameter('Double_t')('t_tot', t_tot))

        t_blocks.append(t_blocks_temp)

    print('Adding tags')
    t_trig = add_tags(t0, hodo_times, trange)
    n_trig = add_tags(tn, hodo_times, trange)

    outfile.Write()
    print("Wrote to", args.out)
    outfile.Close()
    
