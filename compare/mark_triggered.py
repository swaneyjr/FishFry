#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

import ROOT as r

def find_cut(t0, hodo_times, thresh=0):  
    print('Computing diffs')
    alldiffs = []
    for i,evt in enumerate(t0):
        print('{} / {}'.format(i+1, t0.GetEntries()), end='\r')
        if max(evt.cal) >= thresh:
            for th in hodo_times.values():
                diffs = evt.t_adj - th
                alldiffs.append(diffs[np.abs(diffs) < 1000])
    
    alldiffs = np.hstack(alldiffs)
    
    plt.hist(alldiffs, bins=200, range=(-1000, 1000))
    plt.xlabel(r'$\Delta t$')
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
    parser.add_argument('--pfile', required=True, help='phone ROOT file')
    parser.add_argument('--hfile', required=True, help='hodoscope .npz file')
    parser.add_argument('--dt', type=float, default=600, help='Gap (in seconds) above which the hodoscope is considered to be turned off')
    parser.add_argument('--trange', type=int, nargs=2, help='Min and max time difference to trigger')
    parser.add_argument('--thresh', type=int, default=0, help='threshold for calibrated values') 
    parser.add_argument('--out', default='triggered.root', help='Output ROOT file')

    args = parser.parse_args()

    # load phone data
    print('Loading data')
    pfile = r.TFile(args.pfile)
    t0_pre = pfile.Get('triggers')
    tn_pre = pfile.Get('nontriggers')

    pbranches = [b.GetName() for b in t0_pre.GetListOfBranches()]
    if not 't_adj' in pbranches:
        print("ERROR: time corrections not yet set.")
        print("First, use correct_timestamps.py")
        exit()


    # find bounds in time of phone run
    tpmin = 1e15
    tpmax = 0
    print('Iterating...')
    for evt in t0_pre:
        tpmin = min(tpmin, evt.t_adj)
        tpmax = max(tpmax, evt.t_adj)
    for evt in tn_pre:
        tpmin = min(tpmin, evt.t_adj)
        tpmax = max(tpmax, evt.t_adj)
 

    # load hodoscope data and compare time bounds
    hodo = np.load(args.hfile)

    thmin = max(tpmin, hodo['interval_ti'][0])
    thmax = min(tpmax, hodo['interval_tf'][-1])

    print('Copying TTrees')
    r.gROOT.cd()
    t0 = t0_pre.CopyTree('t_adj > {} && t_adj < {}'.format(hodo['interval_ti'][0], hodo['interval_tf'][-1]))
    tn = tn_pre.CopyTree('t_adj > {} && t_adj < {}'.format(hodo['interval_ti'][0], hodo['interval_tf'][-1]))
    print('Done!')
    print()

    a = hodo['millis_a']
    b = hodo['millis_b']
    c = hodo['millis_c']

    dt = (-1,0,1)
 
    hodo_times = {
            'AB': np.unique(np.hstack([np.intersect1d(a,b+t) for t in dt])),
            'BC': np.unique(np.hstack([np.intersect1d(b,c+t) for t in dt])),
            'AC': np.unique(np.hstack([np.intersect1d(c,a+t) for t in dt])),
        }

    trange = args.trange if args.trange else find_cut(t0, hodo_times, args.thresh) 

    # write file with new TTrees
    outfile = r.TFile(args.out, 'recreate')
    
    t_blocks = []
    for c, th in hodo_times.items():

        t_blocks_temp = get_blocks('blocks_{}'.format(c), t0, tn, th, trange)
 
        th = th[(th >= thmin) & (th <= thmax)]
     
        user_info = t_blocks_temp.GetUserInfo()
        user_info.Add(r.TParameter('Double_t')('tolerance', trange[1] - trange[0]))
        user_info.Add(r.TParameter('Double_t')('n_hodo', th.size))
        user_info.Add(r.TParameter('Double_t')('t_tot', thmax - thmin))

        t_blocks.append(t_blocks_temp)

    t_trig = add_tags(t0, hodo_times, trange)
    n_trig = add_tags(tn, hodo_times, trange)

    outfile.Write()
    print("Wrote to", args.out)
    outfile.Close()
    pfile.Close()
    
