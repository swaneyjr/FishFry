#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

from acceptance import random_theta


def mc(x, y, z1, z2, lx, ly, n):

    phi = np.random.uniform(0, 2*np.pi, n)
    theta = random_theta(n)

    x1 = np.abs(x - z1*np.tan(theta)*np.cos(phi)) < lx/2
    y1 = np.abs(y - z1*np.tan(theta)*np.sin(phi)) < ly/2
    x2 = np.abs(x + z2*np.tan(theta)*np.cos(phi)) < lx/2
    y2 = np.abs(y + z2*np.tan(theta)*np.sin(phi)) < ly/2

    return np.logical_and.reduce(np.vstack([x1, y1, x2, y2])).sum()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--z1', type=float, default=4, help='Distance from CMOS to near edge of top LYSO')
    parser.add_argument('--z2', type=float, default=8, help='Distance from CMOS to near edge of bottom LYSO')
    parser.add_argument('--lx', type=float, default=16)
    parser.add_argument('--ly', type=float, default=14)

    parser.add_argument('--nbins', type=int, default=20000, help='Approximate 2D bin total')
    parser.add_argument('--nsamples', type=int, default=50000, help='Number of samples per bin')
    parser.add_argument('--out', help='Name of npz file to save results')


    args = parser.parse_args()
    
    bin_density = args.nbins / (args.lx * args.ly)
    nbins_x = int(bin_density**0.5 * args.lx)
    nbins_y = int(bin_density**0.5 * args.ly)
    xbins = np.linspace(-args.lx/1.5, args.lx/1.5, nbins_x)
    ybins = np.linspace(-args.ly/1.5, args.ly/1.5, nbins_y)

    hgp = np.zeros((nbins_y, nbins_x))
    for iy,y in enumerate(ybins):
        for ix,x in enumerate(xbins):
            print(ix + nbins_x * iy + 1, '/', nbins_x*nbins_y, end='\r')
            hgp[iy, ix] = mc(x, y, args.z1, args.z2, args.lx, args.ly, args.nsamples)
    print('Finished!')

    hgp = hgp / args.nsamples

    if args.out:
        print('Saving to', args.out)
        np.savez(args.out, 
                x=xbins, 
                y=ybins,
                pct_tagged=hgp,
                z=[args.z1,args.z2])

    fig1 = plt.figure(1)
    plt.imshow(hgp, vmin=0, cmap='gnuplot2', extent=[-args.lx/1.5, args.lx/1.5, -args.ly/1.5, args.ly/1.5])
    plt.colorbar()

    cmos = Rectangle((-2.5,-2),5,4,linewidth=1,edgecolor='r',facecolor='none')
    ax1 = fig1.gca()
    ax1.add_patch(cmos)

    fig2 = plt.figure(2)
    ax2 = fig2.gca(projection='3d')
    
    X, Y = np.meshgrid(xbins, ybins)
    surf = ax2.plot_surface(X,Y,hgp, cmap='gnuplot2', linewidth=0, antialiased=False)
    fig2.colorbar(surf)
    plt.show()
