#!/usr/bin/env python3

import numpy as np


def test_bimodal(data, thresh = 2):
    split = np.mean(data)
    upper_dev = np.sqrt(data[data > split].var())
    lower_dev = np.sqrt(data[data <= split].var())
    total_dev = np.sqrt(data.var())
    
    if total_dev / (lower_dev + upper_dev) > thresh:
        return data > split
    
    return np.ones(data.size, dtype=bool)


def recursive_cut(n_sigmas, *data):
    data_new = data
    first = True

    while first or data_new[0].size != data[0].size:
        first = False
        data = data_new
        thresholds_upper = [d.mean() + n_sigmas * np.sqrt(d.var()) for d in data]
        thresholds_lower = [d.mean() - n_sigmas * np.sqrt(d.var()) for d in data]
        cut = np.logical_and.reduce([(d < thresholds_upper[i]) & (d > thresholds_lower[i]) for i,d in enumerate(data)])
        data_new = [d[cut] for d in data]

    rtn = []
    for i in range(len(data)):
        rtn.append(data[i])
        rtn.append(thresholds_upper[i])
    return tuple(rtn)


def find_medians(x, y, cut=None, n_points=100):
    
    if cut is None:
        cut = np.ones(x.size, dtype=bool)

    x_cut = x[cut]
    y_cut = y[cut]

    x_idx = x_cut.argsort()
    x_sorted = x_cut[x_idx]
    y_sorted = y_cut[x_idx]

    x_medians = []
    y_medians = []

    sample_size = x_cut.size / n_points

    for sample in range(n_points):
        istart = sample * x_cut.size // n_points
        iend = (sample+1) * x_cut.size // n_points

        x_sample = x_sorted[istart:iend]
        y_sample = y_sorted[istart:iend]
        x_medians.append(np.median(x_sample))
        y_medians.append(np.median(y_sample))

    # for values outside cut
    x_medians.append(-1)
    y_medians.append(-1)

    # sort with bad pixels last
    sort = np.argsort(np.where(cut, x, x.max()))
    isort = np.zeros(len(sort), dtype=int)
    isort[sort] = np.arange(len(sort))
    median_idx = np.minimum(isort / sample_size, len(x_medians)-1).astype(int)
    
    return np.array(x_medians)[median_idx], np.array(y_medians)[median_idx], x_medians[:-1], y_medians[:-1]


def clean_hotcells(mean, var):
    pk = test_bimodal((var/mean)[mean > 0])
    mean_cut, mean_thresh, var_cut, var_thresh = recursive_cut(5, mean[pk], var[pk])
    mean_medians, var_medians, mean_fit, var_fit = find_medians(mean, var, cut=(mean < mean_thresh) & (var < var_thresh) & pk)
    var_resid = var - var_medians
    resid_cut, resid_thresh = recursive_cut(5, var_resid[var_medians >= 0])

    hotcells = (mean >= mean_thresh) | (var >= var_thresh) | ((var_resid >= resid_thresh) & pk)
    return hotcells, mean_thresh, var_thresh, mean_fit, var_fit


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='')
    parser.add_argument('f', help='Numpy file to load')

    args = parser.parse_args()
    f = np.load(args.f)
    mean = f.f.sum / f.f.num
    var = f.f.ssq / f.f.num - mean**2

    hotcells, mean_thresh, var_thresh, mean_fit, var_fit = clean_hotcells(mean, var)

    mean_cleaned = mean[~hotcells]
    var_cleaned = var[~hotcells]

    print("Hotcells found: %d (%.3f%%)" % (hotcells.sum(), 100*hotcells.sum() / hotcells.size))

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plt.figure(1)
    plt.xlabel('mean')
    plt.ylabel('variance')
    plt.scatter(mean_cleaned[::17], var_cleaned[::17], s=0.1, c=f.f.second[~hotcells][::17], cmap='rainbow', norm=LogNorm())
    plt.plot(mean_fit, var_fit, color='gold')
    plt.colorbar()

    plt.figure(2)
    plt.xlabel('second max')
    plt.hist(f.f.second, bins=np.arange(f.f.second.min(), f.f.second.min() + 80), label='Total', color='black', log=True)
    plt.hist(f.f.second[(mean < mean_thresh) & (var < var_thresh)], bins=np.arange(f.f.second.min(), f.f.second.min() + 80), label='Mean and Var Cuts', color='gray', log=True)
    plt.hist(f.f.second[~hotcells], bins=np.arange(f.f.second.min(), f.f.second.min() + 80), label='Final', color='gold', log=True)
    plt.legend()

    plt.show()

