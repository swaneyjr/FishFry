import numpy as np
import os
import sys

from unpack_trigger import interpret_header

# temporary hack to add pixelstats modules to path
fishfry_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(fishfry_dir, 'pixelstats'))

from lens_shading import load_weights
from hot_pixels import load_hot
from geometry import load_res

class Calibrator:
    def __init__(self, calib_dir, denom=1023):
        self.denom=1023
 
        try:
            wgt = load_weights(calib_dir) 
            self.height, self.width = wgt.shape
            self.weights = (wgt.astype("f4")*self.denom).astype("i4")
        except IOError:
            self.width, self.height = load_res(calib_dir)
            self.weights = np.full((self.height, self.width), 
                    self.denom, dtype='i4')
            print('Could not load lens-shading map')

        try:
            hot = load_hot(calib_dir)
         
            hot_x = hot %  self.width
            hot_y = hot // self.width

            self.weights[hot_y, hot_x] = 0
        except IOError:
            print('Could not load hotcells')

    
    def calibrate_region(self,px,py,region,header):

        width = interpret_header(header,"width")
        height = interpret_header(header,"height")
        if not width == self.width or not height == self.height:
            raise ValueError('Mismatched weight dimensions')

        dx = interpret_header(header,"region_dx")
        dy = interpret_header(header,"region_dy")

        region_size = region.shape[1]
        if region_size != (2*dx+1)*(2*dy+1):
            raise ValueError("inconsistent region size during calibration")

        index = np.arange(region_size)
        offx = index %  (2*dx+1)-dx
        offy = index // (2*dx+1)-dy

        pxs = px.reshape(-1,1) + offx
        pys = py.reshape(-1,1) + offy

        pxs = np.clip(pxs, 0, self.width-1)
        pys = np.clip(pys, 0, self.height-1)
        ws  = self.weights[pys,pxs]
         
        return (region * ws)//self.denom

