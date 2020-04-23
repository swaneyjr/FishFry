import numpy as np
import os

WEIGHTS = None
DENOM = 1023

def _calibrate_load(calib_dir):
    global WEIGHTS

    if not np.any(WEIGHTS):
        f_lens = np.load(os.path.join(calib_dir, 'lens.npz'))
        f_hot  = np.load(os.path.join(calib_dir, 'hot_online.npz'))

        gain = f_lens['lens']
        ds   = f_lens['down_sample']
        hot  = f_hot['hot_list']

        wgt = np.where(gain > 0, 1/gain, 0)
        
        # resample gain array
        short_wgt = (wgt.astype("f4")*DENOM).astype("i4")
        full_wgt  = np.repeat(np.repeat(short_wgt, ds, axis=0), ds, axis=1)
        height, width = full_wgt.shape

        hot_x = hot % width
        hot_y = hot // width

        full_wgt[hot_y, hot_x] = 0

        WEIGHTS = full_wgt

        f_lens.close()
        f_hot.close()

    return WEIGHTS

def calibrate_region(px,py,region,dx,dy,width,height,calib_dir):

    weights = _calibrate_load(calib_dir)

    region_size = region.shape[1]
    if (region_size != (2*dx+1)*(2*dy+1)):
        print("inconsistent region size during calibration")
        exit(0)
    index = np.arange(region_size)
    offx = index %  (2*dx+1)-dx
    offy = index // (2*dx+1)-dy

    pxs = px.reshape(-1,1) + offx
    pys = py.reshape(-1,1) + offy

    pxs = np.clip(pxs, 0, width-1)
    pys = np.clip(pys, 0, height-1)
    ws  = weights[pys,pxs]
     
    return (region * ws)//DENOM
    
