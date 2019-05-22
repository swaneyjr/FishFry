#!/usr/bin/env python

import numpy as np
import ROOT as r

def adjust_times(t0, starts, slopes, offsets):

    # this is easier if we first put the times in order
    isorted = np.argsort(starts)
    starts = starts[isorted]
    slopes = slopes[isorted]
    offsets = offsets[isorted]

    t1 = t0.CloneTree(0)
    t_adj = np.zeros(1, dtype=float)
    t1.Branch('t_adj', t_adj, 't_adj/D')
    
    times = np.array([evt.t for evt in t0])
    
    min_times = np.array([times[times >= ts].min() for ts in starts])
    total_offsets = offsets - min_times * slopes

    for evt in t0:
        # find start time immediately preceding each time
        i = np.argwhere(min_times <= evt.t)[0, 0] 
        t_adj[0] = evt.t * (1 - slopes[i]) - total_offsets[i]
        
        t1.Fill()
       
    return t1


def calibrate_drifts(th, tp, tlim=120):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('text', usetex=True)

    th = np.sort(th)
    tp = np.sort(tp)

    offset_list = []
    slope_list = []

    starts = tp[np.argwhere(np.diff(tp) > 15000).flatten() + 1]
    starts = np.append(tp[0], starts)

    for s in xrange(starts.size):
        alltimes = []
        alldiffs = []
        
        if s+1 < starts.size:
            th_s = th[(th >= starts[s] - tlim) & (th < starts[s+1] - tlim)]
        else:
            th_s = th[th >= starts[s] - tlim]

        for i in xrange(th_s.size):
        
            diffs = tp - th_s[i]
            diffs = diffs[np.abs(diffs) < tlim]
            alldiffs += list(diffs)
            alltimes += [th_s[i] - starts[s]] * diffs.size
 
        alltimes = np.array(alltimes)
        alldiffs = np.array(alldiffs)

        slope_bins = 100
        offset_bins = 100
 
        slope_min = -5e-5
        slope_max = 5e-5

        offset_min = -2
        offset_max = 2

        offset_space = np.linspace(offset_min, offset_max, offset_bins)
        slope_space = np.linspace(slope_min, slope_max, slope_bins)

        slopes = ((alldiffs - offset_space.reshape(-1,1)) / alltimes).flatten()
        offsets = (np.repeat(offset_space.reshape(-1,1), alldiffs.size, axis=1)).flatten()

        hist, binsx, binsy = np.histogram2d(offsets, slopes, bins=(np.linspace(-2, 2, offset_bins+1), np.linspace(slope_min, slope_max, slope_bins+1)))

        #TODO: can we do something fancier here?
        max_bin = np.argmax(hist)
        peak_offset = offset_space[max_bin / offset_bins]
        peak_slope = slope_space[max_bin % slope_bins]

        print "Peak at %.1E, %.1E" % (peak_offset, peak_slope)

        plt.figure(1)
        plt.hist2d(offsets, 3600*slopes, bins=(np.linspace(-2, 2, 101), 3600*np.linspace(slope_min, slope_max)))
        plt.xlabel('Offset (s)')
        plt.ylabel('Slope (s/h)')
        plt.colorbar()

        plt.figure(2)
        plt.scatter(alltimes, alldiffs, s=0.1)
        plt.plot(alltimes, peak_slope * alltimes + peak_offset, color='gold')
        plt.ylim(-10, 10)
        plt.xlabel(r'$t_{hodo} - t_0$')
        plt.ylabel(r'$t_{phone} - t_{hodo}$')

        plt.show()

        offset_list.append(peak_offset)
        slope_list.append(peak_slope)

    np.savez('calib/drifts.npz', starts=starts, slopes=slope_list, offsets=offset_list)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Apply time drift corrections')
    parser.add_argument('pfile', metavar='PFILE', help='ROOT file with FishStand trigger data')
    parser.add_argument('--out', default='time_corrected.root', help='Name of output file')
    parser.add_argument('--calibrate', dest='hfile', help='Calibrate time drifts with a .npz file of hodoscope times')
    args = parser.parse_args()

    pfile = r.TFile(args.pfile)
    t0 = pfile.Get('triggers')

    if args.hfile:
        hodo = np.load(args.hfile)
        th = np.intersect1d(hodo.f.chan_a, hodo.f.chan_b)
        tp = np.array([trig.t for trig in t0])
        calibrate_drifts(th, tp)
    
    try:
        drifts = np.load('calib/drifts.npz')
    except IOError:
        print "No time drift calibration found. Use the --calibrate flag"
        exit()
    
    f = r.TFile(args.out, 'RECREATE')
    t1 = adjust_times(t0, drifts.f.starts, drifts.f.slopes, drifts.f.offsets)
    f.Write()
    f.Close()

    pfile.Close()

