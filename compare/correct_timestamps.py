#!/usr/bin/env python3

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import ROOT as r

rc('text', usetex=True)

def adjust_times(t0, starts, slopes, offsets):

    # this is easier if we first put the times in order
    isorted = np.argsort(starts)
    starts = np.array(starts)[isorted]
    slopes = np.array(slopes)[isorted]
    offsets = np.array(offsets)[isorted]

    t1 = t0.CloneTree(0)
    t_adj = np.zeros(1, dtype=float)
    t1.Branch('t_adj', t_adj, 't_adj/D')
    
    times = np.array([evt.t for evt in t0])
    total_offsets = offsets - starts * slopes
   
    starts_zero = starts.copy()
    starts_zero[0] = 0

    for ievt,evt in enumerate(t0):
        # find start time immediately preceding each time
        print(ievt+1, '/', t0.GetEntries(), end='\r')
        i = np.argwhere(starts_zero <= evt.t).max() 
        t_adj[0] = evt.t * (1 - slopes[i]) - total_offsets[i]
        
        t1.Fill()
       
    return t1


def calibrate_drifts(th, tp, slope_lim, offset_lim, duration=0, tlim=5000): 

    th = np.sort(th)
    tp = np.sort(tp)

    slope_min, slope_max = slope_lim
    offset_min, offset_max = offset_lim

    slope_nbins = 100
    offset_nbins = 100 

    offset_space = np.linspace(offset_min, offset_max, offset_nbins)
    slope_space = np.linspace(slope_min, slope_max, slope_nbins)

    offset_bins = np.linspace(offset_min, offset_max, offset_nbins+1)
    slope_bins = np.linspace(slope_min, slope_max, slope_nbins+1)

    alltimes = []
    alldiffs = []
        
    # keep only relevant hodoscope hits
    th = th[(th >= tp[0] - tlim - offset_max) & (th <= tp[-1] + tlim - offset_min)]

    for i in range(th.size):
        
        diffs = tp - th[i]
        diffs = diffs[np.abs(diffs - np.mean(offset_lim)) < tlim]
        alldiffs += list(diffs)
        alltimes += [th[i] - tp[0]] * diffs.size
 
    print(len(alldiffs))

    alltimes = np.array(alltimes)
    alldiffs = np.array(alldiffs)

    # convert points to slope-offset space
        
    slope_offset = np.zeros((offset_nbins, slope_nbins))
    dt = -duration // 2

    while dt <= duration // 2:
        print(dt, end='\r')
        slopes = ((alldiffs - offset_space.reshape(-1,1) + dt) / alltimes).flatten()
        offsets = (np.repeat(offset_space.reshape(-1,1), alldiffs.size, axis=1)).flatten()

        slope_offset += np.histogram2d(offsets, slopes, bins=(offset_bins, slope_bins))[0]
        dt += (offset_max - offset_min) / offset_nbins

    #TODO: can we do something fancier here?
    max_bin = np.argmax(slope_offset)
    peak_offset = offset_space[max_bin // offset_nbins]
    peak_slope = slope_space[max_bin % slope_nbins] 

    print("Peak at {0:.1E}, {1:.1E}".format(peak_offset, peak_slope))

    plt.figure(figsize=(4,3), tight_layout=True)
    plt.imshow(slope_offset.transpose(), origin='lower', extent=(offset_min, offset_max, 3.6e6*slope_min, 3.6e6*slope_max), aspect='auto')
    plt.xlabel('Offset [ms]')
    plt.ylabel('Slope [ms/h]')
    plt.colorbar()  
 
    return alltimes+tp[0], alldiffs, peak_slope, peak_offset 


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Apply time drift corrections')
    parser.add_argument('pfile', metavar='PFILE', help='ROOT file with FishStand trigger data')
    parser.add_argument('--out', help='Name of output file')
    parser.add_argument('--hodo', dest='hfile', help='Calibrate time drifts with a .npz file of hodoscope times')
    parser.add_argument('--thresh', type=int, default=0, help='Cut on calibrated pixel values')
    parser.add_argument('--calib', help='Path to calibration directory')
    parser.add_argument('--slope_lim', nargs=2, type=float, default=(-100,100), help='Space-separated limits (min max) for slope in ms/hr')
    parser.add_argument('--offset_lim', nargs='+', type=float, default=(-1e3,1e3), help='Space-separated limits (min max) for offset in ms')
    parser.add_argument('--duration', type=int, default=0, help='Frame duration in ms')
    
    parser.add_argument('--splits', nargs='+', type=float, default=[], help='Timestamps of drift resets')
    parser.add_argument('--dt_lim', type=float, default=5000, help='Bounds for time differences to display in ms')


    parser.add_argument('-a', action='store_true', help='Use LYSO A for coincidences')
    parser.add_argument('-b', action='store_true', help='Use LYSO B for coincidences')
    parser.add_argument('-c', action='store_true', help='Use LYSO C for coincidences')

    args = parser.parse_args()

    pfile = r.TFile(args.pfile)
    tn = pfile.Get('nontriggers')  
    t0 = pfile.Get('triggers')


    if not 2*len(args.splits) + 2 == len(args.offset_lim):
        raise ValueError('Incorrect number of arguments passed to --offset_lim')

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

    tp = np.array([trig.t for trig in t0 if trig.max >= args.thresh])

    plt.figure(figsize=(4,3), tight_layout=True)
    ax1 = plt.gca()
    endpoints = [tp[0]] + list(args.splits) + [tp[-1] + 1]
    
    alltimes = []
    alldiffs = []
    starts = []
    slopes = []
    offsets = []
    offset_mean = np.mean(args.offset_lim[:2])
    
    for tmin, tmax in zip(endpoints[:-1], endpoints[1:]):
        tp_interval = tp[(tp >= tmin) & (tp < tmax)]
        th_interval = th[(th >= tmin) & (th < tmax)]

        slope_min = args.slope_lim[0] / 3.6e6
        slope_max = args.slope_lim[1] / 3.6e6

        offset_min = args.offset_lim[0]
        offset_max = args.offset_lim[1]
        args.offset_lim = args.offset_lim[2:]

        times, diffs, slope, offset = calibrate_drifts(th_interval, 
                tp_interval, 
                slope_lim=(slope_min, slope_max), 
                offset_lim=(offset_min, offset_max), 
                duration=args.duration,
                tlim=args.dt_lim)

        alltimes += list(times)
        alldiffs += list(diffs)
        starts.append(tp_interval.min())
        slopes.append(slope)
        offsets.append(offset)
    
        t_off = tp_interval.min()
        t = np.linspace(tmin-t_off, tmax-t_off, 25)
        bound = args.duration / 2

        ax1.plot((t+t_off-tp.min())/1000, slope*t + offset - bound, '--', color='gold')
        ax1.plot((t+t_off-tp.min())/1000, slope*t + offset + bound, '--', color='gold')
    
    ax1.scatter((np.array(alltimes)-tp.min())/1000, alldiffs, s=0.1)
    ax1.set_ylim(offset_mean - args.dt_lim, offset_mean + args.dt_lim)
    ax1.set_xlabel('Time of scintillator tag [s]', fontsize=11)
    ax1.set_ylabel(r'$\Delta t$ (CMOS, scintillator) [ms]', fontsize=11)
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(-5,5))
    plt.show()

    if args.out:
        if args.calib:
            fdrift = os.path.join(args.calib, 'drifts.npz')
            np.savez(fdrift, starts=starts, slopes=slopes, offsets=offsets)
 
        print('Writing triggers')
        f = r.TFile(args.out, 'RECREATE')
        t1 = adjust_times(t0, starts, slopes, offsets)

        print('Writing non-triggers')
        tn1 = adjust_times(tn, starts, slopes, offsets)

        user_info = t1.GetUserInfo()
        user_info.Add(r.TParameter('Int_t')('t_frame', args.duration)) 
    
        f.Write()
        f.Close()

    pfile.Close()

