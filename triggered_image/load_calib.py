#!/usr/bin/env python3

import numpy as np
import gzip

def load_header(f):

    hsize   = np.fromfile(f, dtype='>i4', count=1)[0]
    version = np.fromfile(f, dtype='>i4', count=1)[0]
    res_x   = np.fromfile(f, dtype='>i4', count=1)[0]
    res_y   = np.fromfile(f, dtype='>i4', count=1)[0]
    iso     = np.fromfile(f, dtype='>i4', count=1)[0]
    exp     = np.fromfile(f, dtype='>i4', count=1)[0]

    # get sample_frac values tested
    n_vals  = np.fromfile(f, dtype='>i4', count=1)[0]
    s_fracs = np.fromfile(f, dtype='>f8', count=n_vals)

    return res_x, res_y, iso, exp, s_fracs

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Output calibration data as a .npz file')
    parser.add_argument('--in', dest='infiles', nargs='+', help='.dat files created with "trigger_calib" analysis')
    parser.add_argument('--out', required=True, help='.npz file to output')
    parser.add_argument('-H', '--histogram', action='store_true', help='display histogram of thresholds')
    args = parser.parse_args()

    thresholds = []
    for fname in args.infiles:
        f = gzip.open(fname) if fname.endswith('gz') else open(fname)
        
        res_x, res_y, iso, exp, s_fracs = load_header(f)
        thresholds.append(np.fromfile(f, dtype='>f8'))

        f.close()

    thresholds = np.hstack(thresholds).reshape(-1, s_fracs.size).transpose()

    np.savez(args.out, \
            res_x=res_x, \
            res_y=res_y, \
            iso=iso, \
            exposure=exp, \
            sample_frac = s_fracs, \
            thresholds = thresholds)

    print(thresholds)

    if args.histogram:
        import matplotlib.pyplot as plt
        for ifrac, frac in enumerate(s_fracs):
            plt.figure()
            plt.title('Threshold by image for sample_frac={}'.format(frac))
            plt.hist(thresholds[ifrac], bins=25)
            plt.xlabel('threshold')
            plt.show()
