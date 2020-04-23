#!/usr/bin/env python3

import os
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser('Add hotcells from "hot_offline.npz" into "hot_online.npz"')
parser.add_argument('--calib', default='calib', help='directory for calibration files')
args = parser.parse_args()

fname_off = os.path.join(args.calib, 'hot_offline.npz')
fname_on  = os.path.join(args.calib, 'hot_online.npz')

f_off = np.load(fname_off)
f_on  = np.load(fname_on)

hot_off = f_off['hot_list']
hot_on  = f_on['hot_list']

f_off.close()
f_on.close()

print('Offline:', hot_off.size, 'pixels')
print('Online:', hot_on.size, 'pixels')

if np.any(np.intersect1d(hot_off, hot_on)):
    print('Error: Non-overlapping hotcell lists...')
    print('Exiting.')
    exit(1)

np.savez(fname_on, hot_list=np.sort(np.hstack([hot_off, hot_on])))
print('Saved combined hotcells to "hot_online.npz"')

