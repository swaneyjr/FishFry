#!/usr/bin/env python3

import numpy as np
import argparse
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import pathlib
import os
from time import time

def cluster(x, y, val, threshold, clusteringOption=1):

    # create clusters
    xy = np.column_stack((x,y)) 
        
    if clusteringOption == 1:
        clustering = DBSCAN(eps = threshold, min_samples = 1) 
    elif clusteringOption == 2:
        clustering = AgglomerativeClustering(n_clusters = None, compute_full_tree = True, distance_threshold = threshold)

    clustering.fit(xy)

    ordered_indices = np.argsort(clustering.labels_)
    ordered_labels = clustering.labels_[ordered_indices]

    # split into groups
    diff = np.diff(ordered_labels)
    locations_to_split = (np.argwhere(diff != 0) + 1).flatten()

    groups = np.array_split(ordered_indices, locations_to_split)
            
    # calculates locations of groups based on the greatest luminosity
    group_location_indices = [gp[np.argmax(val[gp])] for gp in groups]
    highest_lum_x = x[group_location_indices]
    highest_lum_y = y[group_location_indices]
        
    # calculates locations of groups based on a weighted average
    xyval = xy * val.reshape(-1,1)
    avg_x, avg_y = np.array([xyval[gp].sum(axis=0) / val[gp].sum() for gp in groups]).transpose()

    return highest_lum_x, highest_lum_y, avg_x, avg_y


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('files', type = str, nargs = '+', help = "pick npz files that contain data for clustering")
    parser.add_argument('--thresh', type = float, help = "set distance threshold", required = True)
    parser.add_argument('--cluster', type = int, help = "pick a clustering algorithm.  1: DBSCAN, 2: Agglomerative Clustering", default = 1)
    parser.add_argument('--out', type = str, default='cluster', help = 'sets directory for .npz files to be saved into.')

    args = parser.parse_args()

    threshold = args.thresh


    for i,fname in enumerate(args.files):
        print('{} / {}'.format(i+1, len(args.files)), end='\r')


        f = np.load(fname)
        x = f['x']
        y = f['y']
        val = f['val']
        f.close()

        max_x, max_y, avg_x, avg_y = cluster(x, y, val, args.thresh, args.cluster)

        inpath = pathlib.PurePath(os.path.realpath(fname))
        outdir = pathlib.Path(inpath.parent.parent.joinpath(args.out))
        if not outdir.is_dir():
            outdir.mkdir(parents = True)

        outbase = inpath
        while outbase.suffix:
            outbase = pathlib.PurePath(outbase.stem)
        outbase = outbase.name
    
        outfile = str(outdir.joinpath(outbase))
            
        np.savez(outfile, max_x=max_x, max_y=max_y, avg_x=avg_x, avg_y=avg_y)


