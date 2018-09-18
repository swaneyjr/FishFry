import numpy as np

header_map = {
    "version"       : 0,
    "images"        : 1,
    "width"         : 2,
    "height"        : 3,
    "sens"          : 4,
    "exposure"      : 5,
    "hist_prescale" : 6,
    "hist_size"     : 7,
}

def unpack_all(filename):
    with open(filename) as f:
        hsize       = np.fromfile(f,dtype=">i4",count=1)[0]
        version     = np.fromfile(f,dtype=">i4",count=1)[0]
        header      = np.fromfile(f,dtype=">i4",count=hsize)
        header      = np.insert(header, 0, version)
        num         = interpret_header(header, "hist_size")
        hist_uncal  = np.fromfile(f,dtype=">i4",count=num)
        hist_nohot  = np.fromfile(f,dtype=">i4",count=num)
        hist_calib  = np.fromfile(f,dtype=">i4",count=num)
        parity_a = parity_b = parity_c = 0
        for i in range(hist_uncal.size):
            parity_a = parity_a ^ hist_uncal[i]
            parity_b = parity_b ^ hist_nohot[i]
            parity_c = parity_c ^ hist_calib[i]
        if ((parity_a != 0) or (parity_b != 0) or (parity_c != 0)):
            print "ERROR: histogram parity bits show data corruption present.\n"
            exit(0)
        hist_uncal = hist_uncal[:-1]
        hist_nohot = hist_nohot[:-1]
        hist_calib = hist_calib[:-1]


        return header,hist_uncal,hist_nohot,hist_calib

def unpack_header(filename):
    header,hist_uncal,hit_nohot,hist_calib = unpack_all(filename)
    return header

def interpret_header(header, param):    
    if param in header_map:
        return header[header_map[param]]
    else:
        print "ERROR:  invalid parameter ", param
        exit(0)

def show_header(header):    
    hsize       = header.size
    print "additional header size:         ", hsize-1
    print "version:                        ", interpret_header(header, "version")
    print "images:                         ", interpret_header(header, "images")
    print "width:                          ", interpret_header(header, "width")
    print "height:                         ", interpret_header(header, "height")
    print "sensitivity:                    ", interpret_header(header, "sens")
    print "exposure:                       ", interpret_header(header, "exposure")
    print "hist_prescale:                  ", interpret_header(header, "hist_prescale")
    print "hist_size:                      ", interpret_header(header, "hist_size")
