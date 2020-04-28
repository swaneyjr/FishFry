#!/usr/bin/env python3

import sys
import os
from unpack import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from geometry import load_res

import argparse

def load_dark(calib):
    width, height = load_res(calib)
    fdark = np.load(os.path.join(calib, 'dark.npz'))

    dark_x = fdark['dark_x']
    dark_y = fdark['dark_y']

    px = fdark['px']
    py = fdark['py']

    fdark.close()

    dark = np.zeros(width*height, dtype=bool)
    
    index = np.arange(width*height)
    x = index % width
    y = index // width
 
    for ix, iy in zip(dark_x, dark_y):
        dark[(x % px == ix) & (y % py == iy)] = True 
    
    return dark

def find_dark(px, py, thresh=None, calib='calib', commit=False):

    # load the image geometry from the calibrations:
    width, height = load_res(calib)
    fgain = np.load(os.path.join(calib, 'gain.npz'))
    gain = fgain.f.gain.reshape(height, width)
    fgain.close()

    # crop to fit the dark pixel period
    crop_x = (width // px - 2) * px
    crop_y = (height // py - 2) * py

    idxx = (width - crop_x) // 2
    idxy = (height - crop_y) // 2

    gain_crop = gain[idxy:-idxy, idxx:-idxx]
    gain_ds = gain_crop.reshape(crop_y//py, py, crop_x//px, px)
    
    # normalize and center at 0
    gain_norm = np.nanmean(gain_ds, axis=(1,3))
    gain_ds /= gain_norm.reshape(crop_y//py, 1, crop_x//px, 1)
    gain_ds -= 1

    gain_mean = np.nanmean(gain_ds, axis=(0,2))
    gain_std = np.nanstd(gain_ds, axis=(0,2))

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.imshow(gain_mean, cmap='coolwarm')
    plt.title('Mean')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(gain_std, cmap='coolwarm')
    plt.title('Standard deviation')
    plt.colorbar()
    plt.show()

    if not thresh: return

    dark_y, dark_x = np.argwhere(gain_mean < -thresh).transpose()
    dark_x = (dark_x + idxx) % px
    dark_y = (dark_y + idxy) % py

    print(dark_x.size, 'dark pixels found')

    if commit:
        outname = os.path.join(calib, 'dark.npz')
        print('saving to', outname)
        np.savez(outname,
                dark_x=dark_x,
                dark_y=dark_y,
                px=px,
                py=py)
    
    
if __name__ == "__main__":
    example_text = '''examples:

    ./dark_pixels.py data/combined/dark_50000.npz'''
    
    parser = argparse.ArgumentParser(description='Combine multiple pixelstats data files.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--calib', default='calib', help='directory to store calibration files')
    parser.add_argument('--commit',action="store_true", help="save dark pixel map to calibration directory")
    parser.add_argument('--period', nargs=2, metavar=("X","Y"), type=int,help="specify dark pixel pattern period in x and y", default=[64,64])
    parser.add_argument('--thresh', type=float, help='absolute value of deviation for counting dark pixels')
    args = parser.parse_args()

    find_dark(*args.period, calib=args.calib, thresh=args.thresh, commit=args.commit)
