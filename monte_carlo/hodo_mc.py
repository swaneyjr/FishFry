#!/usr/bin/env python

import itertools as it

import numpy as np
import matplotlib.pyplot as plt

import ROOT as r

def mc(nf, hodo_rate, noise_rate, eff=1):

    # practically unnecessary check that enough points will be sampled
    while True: 

        t_hodo = np.cumsum(np.random.exponential(1/hodo_rate, int(1.2*nf*hodo_rate)))
        if noise_rate:
            t_noise = np.cumsum(np.random.exponential(1/noise_rate, int(1.2*nf*noise_rate)))
        else:
            t_noise = np.array([nf+1])

        if min(t_hodo.max(), t_noise.max()) > nf: break

    # cut extra frames
    t_hodo = t_hodo[t_hodo < nf]
    t_noise = t_noise[t_noise < nf]
    
    t_trig = t_hodo[np.random.rand(t_hodo.size) < eff]
    f_trig = np.sort(np.hstack([t_trig, t_noise])).astype(int)

    return np.unique(f_trig), t_hodo


def estimate_eff(trig, hodo, nf, t_coinc=1):
    frame_all = np.arange(nf)

    # first get list of untagged frames
    hodo_min = np.searchsorted(hodo-t_coinc, frame_all)
    hodo_max = np.searchsorted(hodo, frame_all)

    # untagged: relative ordering not changed with shift of t_coinc
    untagged = (hodo_min == hodo_max)
    p_noise = untagged[trig].sum() / untagged.sum()

    # now find frame bounds for each hodo and group
    frame_min = np.searchsorted(frame_all, hodo-t_coinc)
    frame_max = np.searchsorted(frame_all, hodo) # exclusive bound
    
    # cluster overlapping intervals
    no_overlap = np.hstack([[True], frame_min[1:] >= frame_max[:-1], [True]])
    tag_min = frame_min[no_overlap[:-1]]
    tag_max = frame_max[no_overlap[1:]]
    n_frames = tag_max - tag_min

    # count number of hodo hits in each interval
    interval_idx = np.arange(frame_min.size+1)
    interval_idx = interval_idx[no_overlap]
    n_hodo = np.diff(interval_idx)
    
    # find which tag intervals are triggered
    tag_lim = np.vstack([tag_min, tag_max]).flatten()
    trig_idx = np.searchsorted(trig, tag_lim).reshape(2,-1)
    tag_triggered = trig_idx[0] != trig_idx[1]
    
    # now analyze the intervals
    
    n_tot = 0
    one_m_eff = 0 
    
    n_vals = np.unique(n_frames)
    m_vals = [3] #np.unique(n_hodo)

    cache = []
    n_M = {}
    fail_M = {}

    # use sqrt(N_trig) as weight
    for M in m_vals:
        n_M[M] = 0
        fail_M[M] = 0 # (1 - eff)**M
        for N in n_vals:

            cut_MN = (n_hodo == M) & (n_frames == N)
            n_MN = cut_MN.sum()

            if not n_MN: continue
            n_M[M] += n_MN

            # save for second pass
            cache.append((M, N, n_MN))  
            
            p_MN = tag_triggered[cut_MN].sum() / n_MN
            fail_MN = (1 - p_MN) / (1 - p_noise)**N
            fail_M[M] += fail_MN * n_MN

            #print('M: {} N: {} tot: {} p: {:.3} eff:{:.3}'.format(M, N, n_MN, p_MN, 1-fail_MN))

        fail_M[M] /= n_M[M]
        one_m_eff_M = (fail_M[M])**(1/M)
        one_m_eff += one_m_eff_M * n_M[M]
        n_tot += n_M[M]

    eff = 1 - one_m_eff / n_tot
    
    # now use the efficiency estimate to calculate the variance
    var1 = 0
    var2 = {M: 0 for M in n_M}
    for M, N, n_MN in cache:
        # calculate p_MN from the full dataset to avoid instabilities
        p_MN = 1 - (1 - eff)**M * (1 - p_noise)**N

        var1 += p_MN / n_MN / (1-p_MN) * (n_MN**(1/M) * n_M[M]**(1-1/M) / M)**2
        var2[M] += n_MN * N

    var = ((1 - eff) / n_tot)**2 * (var1 + p_noise / (1-p_noise) / untagged.size * (sum([n_M[M]**(1-1/M)/M for M,v in var2.items()]))**2)

    return eff, np.sqrt(var)


def estimate_eff_ROOT(trig_times, hodo, nf, t_coinc=1):
    
    t_blocks = r.TTree('MC', 'Independent frame blocks')

    tmin = -t_coinc
    tmax = 0

    t_ = np.array([0], dtype=int)
    M_ = np.array([0], dtype=int)
    N_ = np.array([0], dtype=int)
    pass_ = np.array([False], dtype=bool)

    t_blocks.Branch('t', t_, 't/l')
    t_blocks.Branch('M', M_, 'M/i')
    t_blocks.Branch('N', N_, 'N/i')
    t_blocks.Branch('pass', pass_, 'pass/O')

    nontrig_times = np.delete(np.arange(nf), trig_times)

    trig_times = list(trig_times) + [np.inf]
    nontrig_times = list(nontrig_times) + [np.inf]
    th = np.hstack([[-np.inf], hodo, [np.inf]])
    
    trig_idx = 0
    nontrig_idx = 0
    hodo_idx = 0 

    trig_entries = len(trig_times) - 1
    nontrig_entries = len(nontrig_times) - 1
    hodo_entries = len(th) 

    # get first event, regardless of source
    triggered = trig_times[0] < nontrig_times[0]
    if triggered:
        evt = trig_times[trig_idx]
        trig_idx += 1
    else:
        evt = nontrig_times[nontrig_idx]
        nontrig_idx += 1

    trig_t_next = trig_times[trig_idx]
    nontrig_t_next = nontrig_times[nontrig_idx]

    t_[0] = evt

    while True: 

        # first decide whether to add to existing block 
        # or start a new one
        if evt - th[hodo_idx] >= tmax:

            t_blocks.Fill()
            
            t_[0] = evt
            M_[0] = 0 
            N_[0] = 0
            pass_[0] = False

        while evt - th[hodo_idx+1] >= tmin:
            M_[0] += 1
            hodo_idx += 1

        # update current block
        N_[0] += 1
        if triggered:
            pass_[0] = True

        # load next timestamp
        if trig_idx == trig_entries and nontrig_idx == nontrig_entries:
            # all timestamps have been iterated through
            t_blocks.Fill()
            break

        # check whether to draw next event from triggered
        # or nontriggered datasets
        triggered = trig_t_next < nontrig_t_next

        if triggered:
            evt = trig_times[trig_idx]
            trig_idx += 1
            trig_t_next = trig_times[trig_idx]
        else: 
            evt = nontrig_times[nontrig_idx]
            nontrig_idx += 1
            nontrig_t_next = nontrig_times[nontrig_idx]

    # now analyze the toy tree

    hist_MN = {}

    for iblk, blk in enumerate(t_blocks):
        cmax = getattr(blk, 'pass')
        
        idx_MN = (blk.M, blk.N)
        if not idx_MN in hist_MN:
            hist_MN[idx_MN] = np.zeros(2, dtype=int)
        
        hist_MN[idx_MN][cmax] += 1
    

    # now extract efficiency
    cum_MN = {MN: np.cumsum(h[::-1])[::-1][1:] for MN, h in hist_MN.items()}
    n_MN = {MN: hist.sum() for MN, hist in hist_MN.items()}
    p_MN = {MN: cum_MN[MN] / n_MN[MN] for MN in cum_MN}
    p_noise = p_MN[(0,1)]
    n_untagged = hist_MN[(0,1)].sum()

    one_m_eff = 0
    n_tot = 0
    cache = []
    for MN, cum in cum_MN.items():
        #print('({},{}): {} / {}\t\t'.format(*MN, cum[0], n_MN[MN]))
        
        M,N = MN
        if M != 1: continue
    
        one_m_eff_MN = (1 - p_MN[MN]) / (1 - p_noise)**N

        # weighted average
        one_m_eff += n_MN[MN] * one_m_eff_MN
        n_tot += n_MN[MN]

        # save for second pass
        cache.append((N, n_MN[MN]))

    eff_tot = 1 - one_m_eff / n_tot

    var1 = 0
    var2 = 0

    for N, n in cache:
        # calculate p_MN from the full dataset to avoid instabilities
        p = 1 - (1 - eff_tot) * (1 - p_noise)**N

        var1 += p * n / (1-p)
        var2 += n * N

    var_tot = ((1 - eff_tot) / n_tot)**2 * (var1 + p_noise / (1-p_noise) / n_untagged * var2**2)
    
    return eff_tot[0], np.sqrt(var_tot)[0]


def estimate_eff_old(trig, hodo, nf, t_coinc=1):

    hodo_min = hodo - t_coinc
    hodo_lim = np.vstack([hodo_min, hodo]).transpose().flatten()
    
    # check for overlapping intervals
    no_overlap = np.hstack([[True], np.diff(hodo_lim) > 0, [True]])
    hodo_lim = hodo_lim[no_overlap[1:] & no_overlap[:-1]]

    frame_all = np.arange(nf) 
    
    indices = np.searchsorted(hodo_lim, frame_all)
    tagged_all = (indices % 2 == 1)
    tagged = tagged_all[trig]

    nh = hodo.size

    n = -nf * np.log(1 - trig.size / nf)
    untagged = nf - tagged_all.sum()
    nt = n + untagged * np.log(1 - np.logical_not(tagged).sum() / untagged)

    hodo_factor = np.exp(nh / nf * t_coinc)
    nc = nt * hodo_factor - n*(hodo_factor-1)

    eff = nc / hodo.size

    hodo_err = (n - (n-nt) * (1 - nh / nf * t_coinc) * hodo_factor) / nh**(3/2)
    n_err = 0
    err = np.sqrt(hodo_err**2 + n_err**2)

    return eff, err
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--hodo', required=True, type=float, help='hodoscope rate in frames^-1')
    parser.add_argument('--noise', type=float, default=0, help='noise rate in frames^-1')
    parser.add_argument('--eff', type=float, default=1, help='trigger efficiency for hodoscope hits')
    parser.add_argument('--nf', type=int, default=1000000, help='Number of frames to sample')
    parser.add_argument('--ns', type=int, default=100, help='Number of MC samples')
    parser.add_argument('--window', type=float, default=1, help='Size of coincidence window in frames')
    parser.add_argument('--nbins', type=int, default=80, help='Number of bins to use in histogram')

    args = parser.parse_args()

    eff = []
    err = []
    for i in range(args.ns):
        print(i+1, '/', args.ns, end='\r')
        trig, hodo = mc(args.nf, args.hodo, args.noise, args.eff)
        eff_sample, eff_err = estimate_eff_ROOT(trig, hodo, args.nf, args.window)
        eff.append(eff_sample)
        err.append(eff_err)

    eff_mu = np.mean(eff)
    eff_sigma = np.std(eff)

    print(u'eff = {} \u00B1 {}'.format(eff_mu, eff_sigma/np.sqrt(args.ns)))

    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.hist(eff, bins=args.nbins)
    plt.title(r'Efficiency reconstruction for $\epsilon = {}$'.format(args.eff))
    plt.xlabel('Efficiency estimate')
    plt.ylabel('Frequency')

    plt.subplot(122)
    poll_distr = (np.array(eff) - args.eff) / np.array(err)
    _, bins, _ = plt.hist(poll_distr, bins=args.nbins, color='orange', label='Data')
    bin_width = bins[1] - bins[0]
    lim = max(np.abs(poll_distr))
    x = np.linspace(-lim, lim, 100)
    plt.plot(x, len(eff) * bin_width / np.sqrt(2*np.pi) * np.exp(-x**2/2), label='Standard normal')
    
    plt.title(r'Pull distribution for $\epsilon = {}$'.format(args.eff))
    plt.xlabel('Standard error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

