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
        nbins       = interpret_header(header, "hist_size")

        hists = np.fromfile(f,dtype=">u8")
        
        if not hists.size == 3*nbins:
            raise ValueError('Data corruption: invalid file length')
        
        hist_raw  = hists[:nbins]
        hist_hot  = hists[nbins:2*nbins]
        hist_cal  = hists[-nbins:]

        return header,hist_raw,hist_hot,hist_cal

def unpack_header(filename):
    header, _, _, _ = unpack_all(filename)
    return header

def interpret_header(header, param):    
    if param in header_map:
        return header[header_map[param]]
    else:
        print("ERROR:  invalid parameter ", param)
        exit(0)

def show_header(header):    
    hsize       = header.size
    print("header size:     ", hsize-1)
    print("version:         ", interpret_header(header, "version"))
    print("images:          ", interpret_header(header, "images"))
    print("width:           ", interpret_header(header, "width"))
    print("height:          ", interpret_header(header, "height"))
    print("sensitivity:     ", interpret_header(header, "sens"))
    print("exposure:        ", interpret_header(header, "exposure"))
    print("hist_prescale:   ", interpret_header(header, "hist_prescale"))
    print("hist_size:       ", interpret_header(header, "hist_size"))

