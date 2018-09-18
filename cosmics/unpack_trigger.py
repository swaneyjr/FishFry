import numpy as np

header_map = {
    "version"     : 0,
    "width"       : 1,
    "height"      : 2,
    "sens"        : 3,
    "exposure"    : 4,
    "hot_hash"    : 5,
    "wgt_hash"    : 6,
    "region_dx"   : 7,
    "region_dy"   : 8,
    "denom"       : 9,
    "num_zerobias": 10,
    "num_thresh"  : 11
}

def unpack_all(filename):
    with open(filename) as f:
        hsize       = np.fromfile(f,dtype=">i4",count=1)[0]
        version     = np.fromfile(f,dtype=">i4",count=1)[0]
        header      = np.fromfile(f,dtype=">i4",count=hsize)
        header      = np.insert(header, 0, version)
        num_thresh  = interpret_header(header, "num_thresh")
        threshold   = np.fromfile(f,dtype=">i4",count=num_thresh)
        prescale    = np.fromfile(f,dtype=">i4",count=num_thresh)
        region_dx = interpret_header(header, "region_dx")
        region_dy = interpret_header(header, "region_dy")
        region_size   = (2*region_dx+1)*(2*region_dy+1)        
        region_buffer = region_size + 3 
        px        = np.array([],dtype="u2")
        py        = np.array([],dtype="u2")
        highest   = np.array([],dtype="u2")
        region    = np.array([],dtype="u2")
        timestamp = np.array([],dtype="u4")
        millistamp = np.array([],dtype="u4")
        dropped = 0
        images = 0
        while(1):
            ts = np.fromfile(f,dtype=">i8",count=1)[0]
            if (ts == 0):
                break;
            ms = np.fromfile(f,dtype=">i8",count=1)[0]
            images = images+1
            num_region = np.fromfile(f,dtype=">i4",count=1)[0]
            dropped   += np.fromfile(f,dtype=">i4",count=1)[0]
            region_data = np.fromfile(f,dtype=">i2",count=num_region*region_buffer)
            index = np.arange(num_region*region_buffer)
            px_mask     = ((index%region_buffer) == 0)
            py_mask     = ((index%region_buffer) == 1)
            h_mask     = ((index%region_buffer) == 2)
            region_mask = (px_mask == False) & (py_mask == False) & (h_mask == False)
            px      = np.append(px, region_data[px_mask])
            py      = np.append(py, region_data[py_mask])
            highest = np.append(highest, region_data[h_mask])
            region  = np.append(region, region_data[region_mask])
            timestamp = np.append(timestamp, np.full(num_region, ts))
            millistamp = np.append(millistamp, np.full(num_region, ms))
        region = np.reshape(region,(-1,region_size))
        header = np.append(header, threshold)
        header = np.append(header, prescale)
        return header,px,py,highest,region,timestamp,millistamp,images,dropped

def unpack_header(filename):
    header,px,py,highest,region,images,dropped = unpack_all(filename)
    return header

def interpret_header(header, param):    
    if param in header_map:
        return header[header_map[param]]
    else:
        print "ERROR:  invalid parameter ", param
        exit(0)

def get_trigger(header):
    num_thresh = interpret_header(header, "num_thresh")
    threshold = header[-2*num_thresh:-num_thresh]
    prescale  = header[-num_thresh:]
    return threshold, prescale

def show_header(header):    
    num_thresh = interpret_header(header, "num_thresh")
    denom      = interpret_header(header, "denom")
    hsize       = header.size-2*num_thresh
    print "additional header size:         ", hsize-1
    print "version:                        ", interpret_header(header, "version")
    print "width:                          ", interpret_header(header, "width")
    print "height:                         ", interpret_header(header, "height")
    print "sensitivity:                    ", interpret_header(header, "sens")
    print "exposure:                       ", interpret_header(header, "exposure")
    print "hot_hash:                       ", interpret_header(header, "hot_hash")
    print "wgt_hash:                       ", interpret_header(header, "wgt_hash")
    print "region_dx:                      ", interpret_header(header, "region_dx")
    print "region_dy:                      ", interpret_header(header, "region_dy")
    print "denom:                          ", interpret_header(header, "denom")
    print "num_zerobias:                   ", interpret_header(header, "num_zerobias")
    print "num_thresh:                     ", num_thresh
    threshold, prescale = get_trigger(header)
    print "thresholds:                     ", threshold
    print "prescales:                      ", prescale
 
    


    
