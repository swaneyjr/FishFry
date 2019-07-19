import numpy as np


def _calibrate_load():
     try:
         weights = np.load("calib/weight.npz");
     except:
         print("calib/weight.npz does not exist.")
         return

     weight = weights["wgt"]
     denom = weights["denom"]

     return weight,denom

def calibrate_region(px,py,region,dx,dy,res_x,res_y):

     weight,denom = _calibrate_load()

     region_size = region.shape[1]
     if (region_size != (2*dx+1)*(2*dy+1)):
          print("inconsistent region size during calibration")
          exit(0)
     index = np.arange(region_size)
     offx = index %  (2*dx+1)-dx
     offy = index // (2*dx+1)-dy

     pxs = np.array([offx + x for x in px])
     pys = np.array([offy + y for y in py])

     pxs = np.clip(pxs, 0, res_x-1)
     pys = np.clip(pys, 0, res_y-1)
     ws  = weight[pys,pxs]
     
     return (region * ws)/denom

def calibrate_hot(px,py):
     weight,denom = _calibrate_load()
     hot = (weight[px,py] == 0.0)     
     return hot

    
