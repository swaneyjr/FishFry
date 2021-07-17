#!/usr/bin/env python3

import sys
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import ROOT as r

Pixel = namedtuple('Pixel', ['x', 'y', 'raw', 'cal'])

def cluster(pixels, algorithm):
    
    data = np.array([[x, y, raw, cal] for x, y, raw, cal in zip(pixels.x, pixels.y, pixels.raw, pixels.cal)])

    # fit xy
    algorithm.fit(np.array(data)[:,:2])

    ordered_indices = np.argsort(algorithm.labels_)
    ordered_labels = algorithm.labels_[ordered_indices]

    diff = np.diff(ordered_labels)
    split_idx = (np.argwhere(diff != 0) + 1).flatten()
    groups = np.array_split(ordered_indices, split_idx)

    return [list(map(Pixel._make, data[gp])) for gp in groups]


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Construct clusters from pixels above threshold')

    parser.add_argument('pfiles', nargs = '+', help = "ROOT files containing pixel data to cluster")
    parser.add_argument('--thresh', default=0, type=int, help='Threshold on calibrated trigger value')
    parser.add_argument('--dist', default=1, type=float, help = "set distance threshold")
    parser.add_argument('-a', '--agglom', action='store_true', help="Use sklearn Agglomerative Clustering algorithm (default DBSCAN)")
    parser.add_argument('-p', '--plot', action='store_true', help='Plot distance distribution')
    parser.add_argument('--out', help = 'Output file name')
    args = parser.parse_args()


    t0 = r.TChain('triggers')
    for pfile in args.pfiles:
        t0.Add(pfile)

    if args.thresh:
        t0_pre = t0
        r.gROOT.cd()
        t0 = t0_pre.CopyTree('cal > {}'.format(args.thresh))
 
    n = t0.GetEntries()

    if args.agglom:
        algorithm = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=args.dist)
    else:
        algorithm = DBSCAN(eps=args.dist, min_samples=1)

    if args.plot:

        bins = np.arange(0, 4001, 40)
        hist = np.zeros(bins.size-1)

        for i,trig in enumerate(t0):
            if len(trig.x) == 2:
                continue
                # do this simply for efficiency
                dx = trig.x[0] - trig.x[1]
                dy = trig.y[0] - trig.y[1]
                ds = np.sqrt(dx**2 + dy**2)

                ds_bin = (int(ds) - 1) // 10
                if ds_bin < bins.size - 1:
                    hist[ds_bin] += 1

            elif len(trig.x) > 2:
                print('{} / {}'.format(i, n), end='\r')
        
                x = np.array([x for x in trig.x])
                y = np.array([y for y in trig.y])

                dx = x - x.reshape(-1,1)
                dy = y - y.reshape(-1,1)

                ds = np.sqrt(dx**2 + dy**2)
                ds_bin = (int(ds.max()) - 1) // 10
                if ds_bin < bins.size - 1:
                    hist[ds_bin] += 1

        plt.hist(bins[:-1]+1, bins=bins, weights=hist, log=True)
        plt.xlabel('Euclidean distance')
        plt.title('Distance between above-threshold pixels in same frame')
        plt.show()

    if args.out:

        outfile = r.TFile(args.out, 'recreate')
        t = t0.CloneTree(0)

        frame_n = np.zeros(1, dtype=int)
        frame_occ = np.zeros(1, dtype=int)

        t.Branch('frame_n', frame_n, 'frame_n/i')
        t.Branch('frame_occ', frame_occ, 'frame_occ/i')

        for i,trig in enumerate(t0):
            print('{} / {}'.format(i+1, t0.GetEntries()), end='\r')

            clusters = cluster(trig, algorithm)
            
            for cl in clusters:

                t.x.clear()
                t.y.clear()
                t.raw.clear()
                t.cal.clear()

                for pix in cl:

                    t.x.push_back(int(pix.x))
                    t.y.push_back(int(pix.y))
                    t.raw.push_back(int(pix.raw))
                    t.cal.push_back(int(pix.cal))

                frame_n[0] = i
                frame_occ[0] = len(clusters)
                
                t.Fill()

        outfile.Write()
        outfile.Close()

