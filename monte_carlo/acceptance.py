#!/usr/bin/env python3

import os
import sys
import itertools as it

import numpy as np
import matplotlib.pyplot as plt

# temporary hack to add pixelstats modules to path
fishfry_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(fishfry_dir, 'pixelstats'))
from geometry import load_res

# in Hz*mm^-2
HORIZ_MUON_FLUX = .000185

LYSO_X = 14.
LYSO_Y = 16.
LYSO_Z = 6.

PIX_XY = 1.12e-3

LYSO_LABELS = ['A', 'B', 'C']

# generates a cos^2 distribution of theta values in [0,2*pi)
def random_theta(n):
    rand = np.random.random(n)
    return np.arccos(rand**(1/4))

def monte_carlo(n, xminus, xplus, yminus, yplus, zminus, zplus, initial=0, border=0, track_thresh=0):
    
    x0 = np.random.uniform(xminus[initial]-border, xplus[initial]+border, n)
    y0 = np.random.uniform(yminus[initial]-border, yplus[initial]+border, n)
    z0 = (zplus[initial] + zminus[initial]) / 2

    # cast to numpy arrays
    xminus = np.array(xminus).reshape(-1,1)
    xplus = np.array(xplus).reshape(-1,1)
    yminus = np.array(yminus).reshape(-1,1)
    yplus = np.array(yplus).reshape(-1,1)
    zminus = np.array(zminus).reshape(-1,1)
    zplus = np.array(zplus).reshape(-1,1)

    # geometrical acceptance of telescope
    theta = random_theta(n)
    tan_theta = np.tan(theta)
    phi = np.random.uniform(0, 2.0 * np.pi, n)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # calculate x and y coordinates at both ends of detectors

    xhi = x0 + (zplus-z0) * tan_theta * cos_phi
    yhi = y0 + (zplus-z0) * tan_theta * sin_phi

    xlo = x0 + (zminus-z0) * tan_theta * cos_phi
    ylo = y0 + (zminus-z0) * tan_theta * sin_phi

    #return ((xlo > xminus) & (xlo < xplus)& (ylo > yminus) & (ylo < yplus)) & ((xhi > xminus) & (xhi < xplus) & (yhi > yminus) & (yhi < yplus))

    # parametrize the track as
    # x(t) = xlo + t * (xhi - xlo)
    # y(t) = ylo + t * (yhi - ylo)
    # 0 <= t <= 1

    t_xminus = (xminus - xlo) / (xhi - xlo)
    t_xplus = (xplus - xlo) / (xhi - xlo)
    t_yminus = (yminus - ylo) / (yhi - ylo)
    t_yplus = (yplus - ylo) / (yhi - ylo)

    # now see if these intervals intersect with the volume
    tx = np.sort([t_xminus, t_xplus], axis=0)
    ty = np.sort([t_yminus, t_yplus], axis=0)

    t0 = np.maximum(np.maximum(tx[0], ty[0]), 0)
    t1 = np.minimum(np.minimum(tx[1], ty[1]), 1)

    track_len = (t1 - t0) * (zplus - zminus) / np.cos(theta)

    return (track_len > np.array(track_thresh).reshape(-1,1)) & (t0 <= 1) & (t1 >= 0)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Find relevant geometrical acceptance factors') 
    parser.add_argument('--phone_z', type=float, default=None, help='Location of CMOS (if present) on z axis in mm')
    parser.add_argument('--lyso_z', required=True, type=float, nargs='+', help='Location of LYSO centers on z axis in mm')
    parser.add_argument('--N', type=int, dest='trials', default=100, help='Number of trials')
    parser.add_argument('--n', type=int, default=100000, help='Number of particles to sample per trial')
    parser.add_argument('--lyso_dxy', type=float, default=0.4, help='Horizontal LYSO uncertainty in mm')
    parser.add_argument('--lyso_dz', type=float, default=0.4, help='Vertical LYSO uncertainty in mm')
    parser.add_argument('--lyso_thresh', type=float, default=0, help='Minimum track length (in mm) within LYSO needed to trigger response')
    parser.add_argument('--phone_dxy', type=float, default=0.4, help='Horizontal phone uncertainty in mm')
    parser.add_argument('--phone_dz', type=float, default=0.4, help='Vertical phone uncertainty in mm')
    parser.add_argument('--rot', action='store_true', help='Long dimension of CMOS corresponds to short end of LYSO')
    parser.add_argument('--calib', default='calib', help='Path to calibration directory')
    parser.add_argument('--commit', action='store_true', help='Save results to calibration directory')
    


    args = parser.parse_args()

    x = []
    y = []
    zplus = []
    zminus = []

    sig_xy = []
    sig_z = []

    track_thresh = []
    
    for z in args.lyso_z:
        x.append(LYSO_X)
        y.append(LYSO_Y)
        zminus.append(z - LYSO_Z / 2)
        zplus.append(z + LYSO_Z / 2)
        
        sig_xy.append(args.lyso_dxy)
        sig_z.append(args.lyso_dz)

        track_thresh.append(args.lyso_thresh)

    if not args.phone_z is None:
        sys.path.insert(1, '../pixelstats')
        from geometry import load_res

        cmos_size = PIX_XY * np.array(load_res(args.calib)) 
        if args.rot:
            x.append(cmos_size[0])
            y.append(cmos_size[1])
        else:
            x.append(cmos_size[1])
            y.append(cmos_size[0])
        zminus.append(args.phone_z-0.001)
        zplus.append(args.phone_z+0.001)

        sig_xy.append(args.phone_dxy)
        sig_z.append(args.phone_dz)

        track_thresh.append(0)

    x = np.array(x)
    y = np.array(y)
    zminus = np.array(zminus)
    zplus = np.array(zplus)
    sig_xy = np.array(sig_xy)
    sig_z = np.array(sig_z) 

    # first do hodoscope acceptances 

    # make the flux wider than the initial detector
    border = LYSO_Z * np.tan(35*np.pi/180)

    lyso_idx = list(range(len(args.lyso_z)))
    pairs = {c for c in map(frozenset, it.product(lyso_idx, repeat=2)) if len(c) == 2}
    p_hodo = {c: [] for c in pairs}
    p_hodo_err = {c: [] for c in pairs}
    hodo_rate = {c: [] for c in pairs}

    for i in lyso_idx:

        print()
        print('LYSO {}'.format(LYSO_LABELS[i]))

        sx = LYSO_X + 2*border
        sy = LYSO_Y + 2*border

        # loop over all other lyso combinations
        lyso_other = lyso_idx.copy()
        lyso_other.remove(i)

        sets = list(it.chain.from_iterable(it.combinations(lyso_other, r) for r in range(1, len(lyso_idx))))
        p_all = [[] for _ in sets]

        for tr in range(args.trials):
            print(tr, '/', args.trials, end='\r')

            dx = np.random.normal(0, sig_xy, x.size)
            dy = np.random.normal(0, sig_xy, x.size)
            dz = np.random.normal(0, sig_z,  x.size)

            hits = monte_carlo(args.n, 
                    -x/2+dx,
                    x/2+dx,
                    -y/2+dy,
                    y/2+dy,
                    zminus+dz, 
                    zplus+dz, 
                    i, border, track_thresh)
            hits_initial = hits.sum(axis=1)[i]
        
            for si, s in enumerate(sets):
            
                hs = hits[list(s) + [i]]
                p = np.logical_and.reduce(hs).sum() / hits_initial
                p_all[si].append(p)
            
        
        # these are independent of geometrical uncertainties
        t_elapsed = args.n / HORIZ_MUON_FLUX / (sx * sy)
        lyso_rate = hits.sum(axis=1)[i] / t_elapsed
                
        print('Simulated elapsed time:  {} min'.format(t_elapsed // 60))
        print('Rate of incidence:       {:.5f} mHz'.format(lyso_rate * 1e3)) 

        for si, s in enumerate(sets):
            s_fmt = [LYSO_LABELS[x] for x in s]
            mu  = np.mean(p_all[si])
            sig = np.std(p_all[si])
            print(u'P({} | {}) = {:.5f} \u00B1 {:.5f}'.format(' & '.join(s_fmt), LYSO_LABELS[i], mu, sig))

            # save hodoscope rate in case we are committing result
            if len(s) == 1:
                k = frozenset([i, s[0]])
                p_hodo[k].append(mu)
                p_hodo_err[k].append(sig)
                hodo_rate[k].append(lyso_rate * mu)

    for c in hodo_rate:
        p_hodo[c] = np.mean(p_hodo[c])
        p_hodo_err[c] = np.mean(p_hodo_err[c]) / np.sqrt(2)
        hodo_rate[c] = np.mean(hodo_rate[c])

    # now handle the phone
    
    if args.phone_z:

        p_phone = {}
        p_phone_err = {}

        print()
        print('PHONE')
  
        sets = list(it.chain.from_iterable(it.combinations(lyso_idx, r) for r in range(1, len(lyso_idx)+1)))
        p_all = [[] for _ in sets]

        for tr in range(args.trials):
            print(tr, '/', args.trials, end='\r')

            dx = np.random.normal(0, sig_xy, x.size)
            dy = np.random.normal(0, sig_xy, x.size)
            dz = np.random.normal(0, sig_z,  x.size)

            hits = monte_carlo(args.n, 
                    -x/2+dx,
                    x/2+dx,
                    -y/2+dy,
                    y/2+dy,
                    zminus+dz, 
                    zplus+dz, 
                    -1)
            hits_initial = hits.sum(axis=1)[-1]

            for si, s in enumerate(sets): 

                hs = hits[list(s) + [-1]]
                p = np.logical_and.reduce(hs).sum() / hits_initial
                p_all[si].append(p)

        t_elapsed = args.n / HORIZ_MUON_FLUX / np.product(cmos_size)
        phone_rate = hits.sum(axis=1)[-1] / t_elapsed

        print('Simulated elapsed time:  {} min'.format(t_elapsed // 60))
        print('Rate of incidence:       {:.5f} mHz'.format(phone_rate*1e3))

        for si, s in enumerate(sets):
            mu  = np.mean(p_all[si])
            sig = np.std(p_all[si])

            if len(s) == 2:
                p_phone[frozenset(s)] = mu
                p_phone_err[frozenset(s)] = sig
            
            s_fmt = [LYSO_LABELS[x] for x in s]
            print(u'P({} | {}) = {:.5f} \u00B1 {:.5f}'.format(' & '.join(s_fmt), 'PHONE', mu, sig))


    if args.commit:
        print()
        print('Saving to acceptance.npz:')

        d = {}

        for s in pairs:
            
            l1,l2 = [LYSO_LABELS[l] for l in sorted(list(s))]

            p_hgp = p_phone[s]
            p_hgp_err = p_phone_err[s]

            p_pgh = p_hgp * np.product(cmos_size) / (LYSO_X*LYSO_Y) / p_hodo[s]
            p_pgh_err = p_pgh * np.sqrt((p_hgp_err / p_hgp)**2 + (p_hodo_err[s] / p_hodo[s])**2) 
 
            print(u'P({} & {} | PHONE) = {:.5f} \u00B1 {:.5f}'.format(l1, l2, p_hgp, p_hgp_err))
            print(u'P(PHONE | {} & {}) = {:.5f} \u00B1 {:.5f}'.format(l1, l2, p_pgh, p_pgh_err))
            print('Hodo rate {}{} = {:.5f} mHz'.format(l1, l2, 1e3*hodo_rate[s]))
            print()

            d['p_hgp_{}{}'.format(l1, l2)] = p_hgp 
            d['p_hgp_err_{}{}'.format(l1, l2)] = p_hgp_err
            d['p_pgh_{}{}'.format(l1, l2)] = p_pgh
            d['p_pgh_err_{}{}'.format(l1, l2)] = p_pgh_err
            d['hodo_acc_{}{}'.format(l1, l2)] = hodo_rate[s] / HORIZ_MUON_FLUX
        
        np.savez(os.path.join(args.calib, 'acceptance.npz'),  **d)

