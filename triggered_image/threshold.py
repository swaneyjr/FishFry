#!/usr/bin/env python3

import rawpy
import numpy as np
import argparse
import pathlib
import gzip


def hotcells(array, hotc): 

    x_res = array.shape[1]
    hotx = hotc % (x_res)
    hoty = hotc // (x_res)

    maskedarray = np.ma.masked_array(array)
    maskedarray[hoty,hotx] = np.ma.masked
    return np.ma.filled(maskedarray,0)

def checkPixels(array, threshold):

    y, x = np.array(np.nonzero(array > threshold))
    val = array[y, x]
   
    return x, y, val 


def loadImage(infile, thresh, hot=None, out=None):
    if infile.endswith('.gz'):
        f = gzip.open(infile)
    else:
        f = open(infile)

    image = rawpy.imread(f)
    array = image.raw_image
    
    if not hotc is None:
        array = hotcells(array, hot)
    
    x, y, val = checkPixels(array, thresh)
    
    image.close()
    f.close()
    
    np.savez(out, x = x, y = y, val = val, thresh=thresh)



parser = argparse.ArgumentParser()
parser.add_argument('files', type = str, nargs = '+', help = "gives location of image")
parser.add_argument('--thresh', type = int, help = 'sets threshold', required = True)
parser.add_argument('--out', type = str, default='thresh', help = 'sets directory for .npz files to be saved into')
parser.add_argument('--hot', type = str, help = 'gives location of npz file for a hot cell mask')
args = parser.parse_args()


if args.hot:
    hotf = np.load(args.hot)
    hotc = hotc['hot']
    hotf.close()
else:
    hotc = None
    

for i,f in enumerate(args.files):
    print('{} / {}'.format(i+1, len(args.files)), end='\r')

    inpath = pathlib.PurePath(os.path.realpath(f))
    outdir = pathlib.Path(inpath.parent.parent.joinpath(args.out))
    if not outdir.is_dir():
        outdir.mkdir(parents = True)

    outbase = inpath
    while outbase.suffix:
        outbase = pathlib.PurePath(outbase.stem)
    outbase = outbase.name
    
    outfile = str(outdir.joinpath(outbase))
 
    loadImage(f, args.thresh, hot=hotc, out=outfile)
    
