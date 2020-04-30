#! /usr/bin/env python3

import numpy as np
import os

def load_res(calib_dir): 
    # load the image geometry:
    try:
        geom = np.load(os.path.join(calib_dir, "res.npz"))
    except:
        print("res.npz not found.  Use dump_header.py")
        exit(1)
    width  = geom["width"]
    height = geom["height"]
    
    return width, height

if __name__ == "__main__":
    print("testing geometry utilities:  ")

    width = 20
    height = 20
    dsx = 10
    dsy = 11

    down, nx, ny = down_sample(width, height, dsx, dsy)

    print("min:  ", np.min(down))
    print("max:  ", np.max(down))
    print("size: ", nx*ny)
    
    for i in range(height):
        print(down[i*width:(i+1)*width])

    width = 5328
    height = 3000
    dsx = 8
    dsy = 8

    down, nx, ny = down_sample(width, height, dsx, dsy)

    print("min:  ", np.min(down))
    print("max:  ", np.max(down))
    print("size: ", nx*ny)
    
    

