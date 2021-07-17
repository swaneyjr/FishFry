#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from argparse import ArgumentParser

#COLZ = ['c','m', 'gold', 'saddlebrown']
COLZ = ['r', 'g', 'b']

parser = ArgumentParser()
parser.add_argument('infiles', nargs='+', help='Input files from efficiency_blocks.py')
parser.add_argument('--dot', nargs='+', default=[], help='Add alternative curves with dotted lines')
parser.add_argument('--labels', nargs='+', help='Labels for each input file')
parser.add_argument('--title', help='Plot title')
parser.add_argument('-x', '--logx', action='store_true', help='Set x log scale')
parser.add_argument('-y', '--logy', action='store_true', help='Set y log scale')
parser.add_argument('--xlim', type=float, nargs=2, help='Threshold range in electrons')
parser.add_argument('--ylim', type=float, nargs=2, help='Efficiency range')
parser.add_argument('--n_across', type=int, default=1, help='Number of image placements per line in paper (used for sizing)')
args = parser.parse_args()

if args.labels and len(args.labels) != len(args.infiles):
    print('Number of labels does not match number of files. Exiting.')
    exit(1)

labels = args.labels if args.labels else [None]*len(args.infiles)

# check if we should include ADC on x axis
gmin=None
same_gain = False # True
for fname in args.infiles:
    f = np.load(fname)
    gmin_new = np.mean(f['thresh']/f['electrons'])
    if not gmin:
        gmin = gmin_new
    elif np.abs(1 - gmin_new / gmin) > 0.03:
        same_gain=False
    f.close()

vert = 4.5 if same_gain else 3.0
if args.n_across == 1:
    figsize = (4.6, 3.0)
elif args.n_across == 2:
    figsize = (3.7, 3.0)
else:
    figsize = (2.8, 3.0) 
plt.figure(figsize=figsize)
if same_gain:
    ax_thresh = plt.gca()
    ax_thresh.set_xlabel('Threshold [DN, calib.]') 

    ax_e = ax_thresh.twiny() 
    ax_e.xaxis.set_ticks_position('bottom')
    ax_e.xaxis.set_label_position('bottom')
    ax_e.spines['bottom'].set_position(('outward', 40))
    ax_e.set_xlabel('Threshold [$e^-$]')

    ax = ax_thresh

else:
    ax_e = plt.gca()
    ax_e.set_xlabel(r'Threshold [$e^-$]')

    ax = ax_e

ax.set_ylabel('Efficiency')

if args.title:
   ax.set_title(args.title)

for i, fname, label in zip(range(len(args.infiles)), args.infiles, labels):
    f = np.load(fname)
    print(label)

    ax_e.plot(f['electrons'], f['eff'], '-', linewidth=1, color=COLZ[i], label=label)
    ax_e.fill_between(f['electrons'], f['eff']-f['err'], f['eff']+f['err'], color=COLZ[i], alpha=0.2, edgecolor=None)

    f.close()

for i, fname in enumerate(args.dot):
    f = np.load(fname) 
    ax_e.plot(f['electrons'], f['eff'], ls='--', linewidth=1, color=COLZ[i])
    ax_e.fill_between(f['electrons'], f['eff']-f['err'], f['eff']+f['err'], color=COLZ[i], alpha=0.2, edgecolor=None)
    f.close()

if args.xlim:
    ax_e.set_xlim(*args.xlim)
    if same_gain:
        ax_thresh.set_xticks(np.arange(0,1024,100))
        ax_thresh.set_xlim(*(np.array(args.xlim)*gmin))

elif same_gain:
    ax_thresh.set_xlim(*(np.array(ax_e.get_xlim()) * gmin))

if args.labels:
    if args.dot:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='Pb'))
        ax_e.legend(handles=handles)
    else:
        ax_e.legend(loc='lower left')
if args.logx:
    ax.semilogx()
    if same_gain:
        ax_e.semilogx()
if args.logy:
    ax.semilogy()
if args.ylim:
    ax.set_ylim(*args.ylim)

plt.tight_layout()
plt.show()
