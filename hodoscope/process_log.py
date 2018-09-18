#! /usr/bin/env python

# dump a header from run data

import sys
import argparse
import re
import datetime
import numpy as np


max_time = None
min_time = None

chan_a = np.array([], dtype=float)
chan_b = np.array([], dtype=float)
chan_c = np.array([], dtype=float)
hits = np.array([], dtype=int)

def init(filename,args):
    global max_time, min_time, ref_time

    if (args.max_time != ""):
        max_time = datetime.datetime.strptime(args.max_time, '%Y-%m-%d %H:%M:%S.%f')
        print "dropping data after heartbeat:  ", max_time

    if (args.min_time != ""):
        min_time = datetime.datetime.strptime(args.min_time, '%Y-%m-%d %H:%M:%S.%f')
        print "dropping data before heartbeat:  ", min_time


def process(filename,args):
    global max_time, min_time, ref_time
    global chan_a, chan_b, chan_c
    global hits

    if (min_time != None):
        acquire = False
    else:
        acquire = True

    a_ard = np.array([], dtype=int)
    b_ard = np.array([], dtype=int)
    c_ard = np.array([], dtype=int)
    h_ard = np.array([], dtype=int)
    h_rpi = np.array([], dtype=float)

    update = np.zeros(4,dtype="int")            

    reference_re = re.compile(r"new log started at\s+(\S+\s+\S+)")
    heartbeat_re = re.compile(r"heartbeat:\s+(\S+\s+\S+)")    
    update_re    = re.compile(r"update  (\S+) (\S+) (\S+) (\S+)")    
    end_re       = re.compile(r"waiting for heartbeat daemon to acknowledge exit...")

    input = open(filename, 'r')
    line = input.readline();
    match = reference_re.match(line)
    if (match == None):
        print "log file format error."
        exit(0)
    
    

    rpi_heartbeat = 0
    ard_heartbeat = 0
    
    print input.readline(),

    epoch = datetime.datetime.utcfromtimestamp(-7*60*60)  # correction for PST
  
    count = -1
    while(1):
        line = input.readline()
        if not line: break
        match = heartbeat_re.match(line)
        if (match != None):            
            date_str = match.group(1)
            # e.g. 2018-09-14 21:38:50.005481
            date = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
            if (max_time != None):
                if (date >= max_time):
                    break
            if (not acquire):
                if (date > min_time):
                    acquire = True
                else:
                    continue
            #print date_str, "-->", date
            delta = date - epoch
            h = delta.total_seconds()
            rpi_heartbeat += 1
            h_rpi = np.append(h_rpi, h)
            continue

        if (not acquire):
            continue

        match = update_re.match(line)
        if (match != None):
            update = np.array(match.group(1,2,3,4),dtype="int")            
            count = np.sum(update)
            ard_heartbeat += update[3]
            hits = np.empty(0)
            continue

        match = end_re.match(line)
        if (match != None):
            break

        # must be a PMT time:
        x = int(line)
        hits = np.append(hits, x)

        if (hits.size == count):
            slices = np.array([np.sum(update[0:i+1]) for i in range(update.size)])
            #print update, slices, hits
            a_ard = np.append(a_ard, hits[0:slices[0]])
            b_ard = np.append(b_ard, hits[slices[0]:slices[1]])
            c_ard = np.append(c_ard, hits[slices[1]:slices[2]])
            h_ard = np.append(h_ard, hits[slices[2]:slices[3]])
            count = -1
            update[3] = 0

    #check for roll-over:
    if (not np.all(h_ard[:-1] < h_ard[1:])):
        print "ERROR:  arduino time roll-over detected.";
        return

    #unfinished updates (reported but not collected)
    unfinished = update[3]
    print "unfinished heartbeats:  ", unfinished
    ard_heartbeat -= unfinished

    #debounce the arduino heartbeat:
    delta =  (h_ard[1:] - h_ard[0:-1])
    bounce = np.array([False])
    bounce = np.append(bounce, np.array((delta < 5000)))
    bounces = np.sum(bounce)
    print "removing", bounces, "hearbeat bounces."
    h_ard = h_ard[(bounce == False)]
    ard_heartbeat -= bounces

    # check for duplicates:
    print "duplicates in channel a:  ", a_ard[a_ard[1:] == a_ard[:-1]]
    print "duplicates in channel b:  ", b_ard[b_ard[1:] == b_ard[:-1]]
    print "duplicates in channel c:  ", c_ard[c_ard[1:] == c_ard[:-1]]

    # remove duplicates:
    a_ard = np.unique(a_ard)
    b_ard = np.unique(b_ard)
    c_ard = np.unique(c_ard)

    print "rpi sent ", rpi_heartbeat, "heartbeats."
    print "h size:  ", h_rpi.size
    print "arduino saw ", ard_heartbeat, "heartbeats after debouncing." 
    print "a size:  ", a_ard.size
    print "b size:  ", b_ard.size
    print "c size:  ", c_ard.size
    print "h size:  ", h_ard.size


    if (h_ard.size > h_rpi.size):
        print "too many arduino heartbeats detected."

    if (h_ard.size < h_rpi.size):
        h_rpi = h_rpi[:h_ard.size]

    chan_a = np.append(chan_a, np.interp(a_ard, h_ard, h_rpi))
    chan_b = np.append(chan_b, np.interp(b_ard, h_ard, h_rpi))
    chan_c = np.append(chan_c, np.interp(c_ard, h_ard, h_rpi))

    if (not np.all(chan_a[:-1] < chan_a[1:])):
        print "ERROR:  calibrated times are not monotonically increasing for channel a..";
        return
    if (not np.all(chan_b[:-1] < chan_b[1:])):
        print "ERROR:  calibrated times are not monotonically increasing for channel b..";
        return
    if (not np.all(chan_c[:-1] < chan_c[1:])):
        print "ERROR:  calibrated times are not monotonically increasing for channel c..";
        return


def analysis(args):
    global ref_time, chan_a, chan_b, chan_c

    start_time = min(np.min(chan_a), np.min(chan_b), np.min(chan_c))
    end_time = max(np.max(chan_a), np.max(chan_b), np.max(chan_c))

    print start_time, " -> ", datetime.datetime.fromtimestamp(start_time)
    print end_time, " -> ", datetime.datetime.fromtimestamp(end_time)


    np.savez(args.out,chan_a=chan_a, chan_b=chan_b, chan_c=chan_c)



if __name__ == "__main__":

    example_text = '''examples:

    ./process_log.py log.txt --out calib.npz'''
    
    parser = argparse.ArgumentParser(description='Combine multiple pixelstats data files.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument('files', metavar='FILE', nargs='+', help='file to process')
    parser.add_argument('--sandbox',action="store_true", help="run trial code")
    parser.add_argument('--out',metavar='OUT', help="output filename OUT",default="out.npz") 
    parser.add_argument('--max_time', help="only consider times before string MAX_TIME",default="") 
    parser.add_argument('--min_time', help="only consider times before string MIN_TIME",default="") 
    
    args = parser.parse_args()


    init(args.files[0], args)

    for filename in args.files:
        process(filename, args)

    analysis(args)


        
