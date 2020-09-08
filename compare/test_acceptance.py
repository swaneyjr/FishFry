#!/usr/bin/env python3

import os
import sys
import itertools as it

import numpy as np
import matplotlib.pyplot as plt

# in Hz*mm^-2
HORIZ_MUON_FLUX = .0067

LYSO_X = 14.
LYSO_Y = 16.
LYSO_Z = 6.

PIX_XY = 1.12e-3

# generates a cos^2 distribution of theta values in [0,2*pi)
def random_theta(n):
    rand = np.random.random(n)
    return np.arccos(rand**(1/4))

def monte_carlo(n, x, y, zplus, zminus, initial=0, source=(LYSO_X,LYSO_Y)):
 
    x_src = source[0] / 2
    y_src = source[1] / 2
    
    x0 = np.random.uniform(-x_src, x_src, n)
    y0 = np.random.uniform(-y_src, y_src, n)
    z0 = (zplus[initial] + zminus[initial]) / 2

    # cast to numpy arrays
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    zplus = np.array(zplus).reshape(-1,1)
    zminus = np.array(zminus).reshape(-1,1)

    # geometrical acceptance of telescope
    tan_theta = np.tan(random_theta(n))
    phi = np.random.uniform(0, 2.0 * np.pi, n)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # calculate x and y coordinates at both ends of detectors

    xplus = x0 + (zplus-z0) * tan_theta * cos_phi
    yplus = y0 + (zplus-z0) * tan_theta * sin_phi

    xminus = x0 + (zminus-z0) * tan_theta * cos_phi
    yminus = y0 + (zminus-z0) * tan_theta * sin_phi

    # parametrize the track as
    # x(t) = xminus + t * (xplus - xminus)
    # y(t) = yminus + t * (yplus - yminus)
    # 0 <= t <= 1

    t_xlo = (-x/2 - xminus) / (xplus - xminus)
    t_xhi = (x/2 - xminus) / (xplus - xminus)
    t_ylo = (-y/2 - yminus) / (yplus - yminus)
    t_yhi = (y/2 - yminus) / (yplus - yminus)

    # now see if these intervals intersect with the volume
    tx = np.sort([t_xlo, t_xhi], axis=0)
    ty = np.sort([t_ylo, t_yhi], axis=0)

    t0 = np.maximum(tx[0], ty[0])
    t1 = np.minimum(tx[1], ty[1])

    return (t0 <= t1) & (t0 <= 1) & (t1 >= 0)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Find relevant geometrical acceptance factors') 
    parser.add_argument('--phone_z', type=float, default=None, help='Location of CMOS (if present) on z axis in mm')
    parser.add_argument('--lyso_z', required=True, type=float, nargs='+', help='Location of LYSO centers on z axis in mm')
    parser.add_argument('--n', type=int, default=1000000, help='Number of particles to sample')
    parser.add_argument('--eff', type=float, default=1., help='Efficiency of each scintillator/PMT')
    parser.add_argument('--uncertainty', dest='sigma', type=float, default=.1, help='Uncertainty in mm of all length measurements')
    parser.add_argument('--calib', help='calibration directory')

    args = parser.parse_args()

    x = []
    y = []
    zplus = []
    zminus = []
    
    for z in args.lyso_z:
        x.append(LYSO_X)
        y.append(LYSO_Y)
        zplus.append(z + LYSO_Z / 2)
        zminus.append(z - LYSO_Z / 2)

    if not args.phone_z is None:
        sys.path.insert(1, '../pixelstats')
        from geometry import load_res

        cmos_size = PIX_XY * np.array(load_res(args.calib)) 
        x.append(cmos_size[0])
        y.append(cmos_size[1])
        zplus.append(args.phone_z)
        zminus.append(args.phone_z)

    # first do hodoscope acceptances
    
    # make the flux wider than the initial detector
    border = LYSO_Z * np.tan(75*np.pi/180)
    lyso_idx = list(range(len(args.lyso_z)))

    for i in lyso_idx:
        sx = LYSO_X + 2*border
        sy = LYSO_Y + 2*border

        hits = monte_carlo(args.n, x, y, zplus, zminus, i, source=(sx, sy))
        hits_initial = hits.sum(axis=1)[i]

        t_elapsed = args.n / HORIZ_MUON_FLUX / (sx * sy)
        hodo_rate = hits.sum(axis=1)[i] / t_elapsed
    
        print()
        print('LYSO {}'.format(i+1))
        print('Simulated elapsed time:  ', int(t_elapsed), 'min')
        print('Rate of incidence:       ', '{:.5f}'.format(hodo_rate), '/ min')
       
        # loop over all other lyso combinations
        lyso_other = lyso_idx.copy()
        lyso_other.remove(i)

        for s in it.chain.from_iterable(it.combinations(lyso_other, r) for r in range(1, len(lyso_idx))):
            
            hs = hits[list(s) + [i]]
            p = np.logical_and.reduce(hs).sum() / hits_initial

            s_fmt = map(lambda x: str(x+1), s)
            print('P({} | {}) = {:.5f}'.format(' & '.join(s_fmt), i+1, p))

    # now handle the phone
    
    if args.phone_z:

        print('PHONE')
        hits = monte_carlo(args.n, x, y, zplus, zminus, -1, source=cmos_size)
        hits_initial = hits.sum(axis=1)[-1]

        t_elapsed = args.n / HORIZ_MUON_FLUX / np.product(cmos_size)
        hodo_rate = hits.sum(axis=1)[-1] / t_elapsed

        print('Simulated elapsed time:  ', int(t_elapsed), 'min')
        print('Rate of incidence:       ', '{:.5f}'.format(hodo_rate), '/ min')

        for s in it.chain.from_iterable(it.combinations(lyso_idx, r) for r in range(1, len(lyso_idx)+1)):

            hs = hits[list(s)]
            p = np.logical_and.reduce(hs).sum() / hits_initial

            s_fmt = map(lambda x: str(x+1), s)
            print('P({} | {}) = {:.5f}'.format(' & '.join(s_fmt), 'PHONE', p))

 
    if args.calib:

        #TODO
        np.savez(os.path.join(args.calib, 'geometry.npz'), 
            p_pgh = p_pgh,
            p_pgh_err = 0, 
            p_hgp = p_hgp,
            p_hgp_err = 0,
            rate = args.eff**2 * hodo_acceptance * muon_flux * np.product(lyso_dims))

