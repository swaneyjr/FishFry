#!/usr/bin/env python3

import numpy as np
import random
from scipy.optimize import brentq

import os
import sys
sys.path.insert(1, '../pixelstats')
from geometry import load_res

# in Hz*mm^-2
muon_flux = 1/(60*100)


# generates a cos^2 distribution of theta values in [0,2*pi)
def random_theta(n):
    rand = np.random.uniform(0, np.pi / 2, n)
    cdf = lambda theta, r: theta + np.sin(2.0 * theta) / 2 - r
    thetas = [brentq(cdf, 0., np.pi/2.0, args=rand[i]) for i in range(n)]
    return np.array(thetas)


def monte_carlo(n, lyso_x, lyso_y, cmos_x, cmos_y, start_gap, end_gap, rot=False):
    start_x = lyso_x
    start_y = lyso_y

    if rot:
        end_x = lyso_y
        end_y = lyso_x
    else:
        end_x = lyso_x
        end_y = lyso_y

    # geometrical acceptance of telescope
    tan_theta = np.tan(random_theta(n))
    phi = np.random.uniform(0, 2.0 * np.pi, n)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi) 

    # points of interest: cmos and closest ends of paddles
    z_vals = np.array([0, start_gap, start_gap + end_gap]).reshape(-1, 1)

    xvals = np.random.uniform(-start_x/2.0, start_x/2.0, n) \
            + z_vals * tan_theta * cos_phi
    yvals = np.random.uniform(-start_y/2.0, start_y/2.0, n) \
            + z_vals * tan_theta * sin_phi
    
    cmos = (np.abs(xvals[1]) < cmos_x/2.0) & (np.abs(yvals[1]) < cmos_y/2.0)
    hodo = (np.abs(xvals[2]) < end_x/2.0) & (np.abs(yvals[2]) < end_y/2.0)


    hodo_acceptance = hodo.sum() / n
    p_cmos_given_hodo = (hodo & cmos).sum() / hodo.sum()

    return hodo_acceptance, p_cmos_given_hodo


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Find relevant geometrical acceptance factors')
    parser.add_argument('--calib', default='calib', help='calibration directory')
    parser.add_argument('--lyso_dims', required=True, nargs=2, type=float, help='"x y" in mm, separated by spaces')
    parser.add_argument('--gap', required=True, type=float, nargs="+", help='Separation between CMOS and LYSO crystals in mm.  Either a single argument for both or two arguments for top and bottom.')
    parser.add_argument('--rot', action='store_true', help='The bottom scintillator is rotated 90 degrees about the z axis with respect to the top and the CMOS')
    parser.add_argument('--pix_size', type=float, default='1.1', help='Side length of a pixel in um')
    parser.add_argument('--n', type=int, default=10000, help='Number of particles to sample')
    parser.add_argument('--eff', type=float, default=1., help='Efficiency of each scintillator/PMT')
    parser.add_argument('--uncertainty', dest='sigma', type=float, default=.2, help='Uncertainty in mm of all length measurements')

    args = parser.parse_args()

    lyso_dims = np.array(args.lyso_dims)
    cmos_size = args.pix_size/1000 * np.array(load_res(args.calib))

    if len(args.gap) == 1:
        gaps = np.array([args.gap, args.gap])
    else:
        gaps = np.array([args.gap])

    hodo_acceptance, p_pgh = monte_carlo(args.n, *lyso_dims, *cmos_size, gaps[0], gaps[1], args.rot)

    # very rough estimate of uncertainty
    _, p_pgh_plus = monte_carlo(args.n, *(lyso_dims-args.sigma), *cmos_size, *(gaps+args.sigma),args.rot)
    _, p_pgh_minus = monte_carlo(args.n, *(lyso_dims+args.sigma), *cmos_size, *(gaps-args.sigma),args.rot)

    p_pgh_err = np.abs(p_pgh_plus - p_pgh_minus) / 4

    p_hgp = args.eff**2 * p_pgh * hodo_acceptance * np.product(lyso_dims) / np.product(cmos_size)
    p_hgp_err = p_hgp * hodo_acceptance * (p_pgh_err / p_pgh)

    np.savez(os.path.join(args.calib, 'geometry.npz'), 
            p_pgh = p_pgh,
            p_pgh_err = p_pgh_err, 
            p_hgp = p_hgp,
            p_hgp_err = p_hgp_err,
            rate = args.eff**2 * hodo_acceptance * muon_flux * np.product(lyso_dims))

