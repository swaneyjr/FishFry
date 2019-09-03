import numpy
import argparse
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import pathlib


def maxValue(place_holder, luminosity):
    highestLumIndice = numpy.argmax(luminosity[place_holder[0]]);
    return highestLumIndice;

def indicesOfMaxLuminosity(place_holders, array, locations):
    
    return array[place_holders[0]][locations[place_holders[0]]]; 

def brightest(groups,luminosity):
    place_holder = numpy.arange(len(luminosity));
    place_holder = place_holder.reshape(len(luminosity),1);
    
    indicesForParticularGroups = numpy.apply_along_axis(maxValue,1, place_holder, luminosity);
    
    indicesOfMaxLuminosityArray = numpy.apply_along_axis(indicesOfMaxLuminosity, 1, place_holder, groups, indicesForParticularGroups);
    
    return indicesOfMaxLuminosityArray;


def average(place_holder, xy, groups):
    return numpy.take(xy, groups[place_holder[0]]).sum() / (groups[place_holder[0]].size);




parser = argparse.ArgumentParser();
parser.add_argument('-dt', type = float, nargs = 1, help = "set distance threshold", required = True);

parser.add_argument('-f', type = str, nargs = '+', help = "pick npz files that contain data for clustering", required = True);

parser.add_argument('-c', type = int, nargs = 1, help = "pick a clustering algorithm.  1: DBSCAN, 2: Agglomerative Clustering", default = [1]);

parser.add_argument('-out', type = str, nargs = 1, help = "select an output file");

args = parser.parse_args();

threshold = args.dt[0];
f = args.f;
clusteringOption = args.c[0];


for files in f:

    x = numpy.load(files)['x'];
    y = numpy.load(files)['y'];
    xy = numpy.column_stack((x,y));
    luminosity = numpy.load(files)['luminosities'];
    
    
    if clusteringOption == 1:
        clustering = DBSCAN(eps = threshold, min_samples = 1);
        clustering.fit(xy); 
    elif clusteringOption == 2:
        clustering = AgglomerativeClustering(n_clusters = None, compute_full_tree = True, distance_threshold = threshold);
        clustering.fit(xy);

    
    ordered_indices = numpy.argsort(clustering.labels_);
    ordered_labels = numpy.sort(clustering.labels_);
    

    diff = numpy.diff(ordered_labels);
    locations_to_split = numpy.argwhere(diff != 0) + 1;

    groups = numpy.array_split(ordered_indices, locations_to_split.flatten());

    luminosity_split = numpy.array_split(numpy.take(luminosity,ordered_indices),locations_to_split.flatten());



    group_location_indices = brightest(groups, luminosity_split);
    
    #calculates locations of groups based on the greatest luminosity
    highest_lum_x = numpy.take(x, group_location_indices.astype('int'));
    highest_lum_y = numpy.take(y, group_location_indices.astype('int'));
            
    
    #calculates locations of groups based on a weighted average
    avg_x = numpy.apply_along_axis(average, 1, numpy.arange(len(groups)).reshape(len(groups),1), x, groups);
    avg_y = numpy.apply_along_axis(average, 1, numpy.arange(len(groups)).reshape(len(groups),1), y, groups);
   

    if args.out:
        
        if args.c[0] == 1:
            outFile = args.out[0] + '/' + files.split('.')[0] + " DBSCAN Clusters.npz";
        elif args.c[0] == 2:
            outFile = args.out[0] + '/' + files.split('.')[0] + " AgglomerativeClustering Clusters.npz";

        Path = pathlib.Path(args.out[0]);
        if not Path.is_dir():
            Path.mkdir(parents = True);

    else:
        if args.c[0] == 1:
            outFile = files.split('.')[0] + " DBSCAN Clusters.npz";
        elif args.c[0] == 2:
            outFile = files.split('.')[0] + " AgglomerativeClustering Clusters.npz";
        
    numpy.savez(outFile, highest_lum_x = highest_lum_x, highest_lum_y = highest_lum_y, avg_x = avg_x, avg_y = avg_y);


