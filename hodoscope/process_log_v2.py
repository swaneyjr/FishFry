#!/usr/bin/env python3

import re
from datetime import datetime

import numpy as np

CHANNELS = ('A', 'B', 'C')

REGEX_TIMESTAMP = re.compile(r'([ABCH]) (\d+)')
REGEX_HEARTBEAT = re.compile(r'heartbeat:\s+(\S+\s+\S+)') 
REGEX_THRESHOLD = re.compile(r'pwm thresholds: (\d+) (\d+) (\d+)')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--in', dest='infiles', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--tmin', type=float, help='Minimum epoch time in ms')
    parser.add_argument('--tmax', type=float, help='Maximum epoch time in ms')
    args = parser.parse_args()

    # input fields in the npz file
    timestamps = {c: [] for c in CHANNELS}
    thresholds = {c: [] for c in CHANNELS}
    millis_rpi = {c: [] for c in CHANNELS}
        
    h_ard = []
    h_rpi = []

    interval_ti = []
    interval_tf = []
    interval_thresh = {c: [] for c in CHANNELS}

    t_base = 0

    for fname in args.infiles:
        print('processing', fname)

        f_in = open(fname, 'r')
        t_micros = {c: [] for c in CHANNELS} # this file's timestamps

        # start/stop a new unbiased threshold interval with heartbeats
        new_thresh = False
        last_heartbeat = 0

        # handle wraparound in microsecond timestamps after 2^32
        last_t = 0 

        # synchronize time bases of heartbeats
        h_returned = True

        # ignore everything before "hodoscope initialized"
        init = False


        for line in f_in:
           
            if line == 'hodoscope initialized\n': 
                init = True
            elif not init: continue
            
            elif m := REGEX_TIMESTAMP.match(line):
                channel = m.group(1)
                t = int(m.group(2))

                # fix wraparound in unsigned long timestamps
                if t < last_t:
                    if t < 1e9 and last_t > 2**32 - 1e9: # 100s edge
                        t_base += 2**32
                last_t = t

                t += t_base 

                if channel == 'H':   

                    # debounce
                    if not h_ard or t - 5e5 > h_ard[-1] and not h_returned:
                        h_ard.append(t)
                    #elif h_returned and t - 1e6 > h_ard[-1]:    
                    #    print('Extra heartbeat at {}!'.format(t-t_base))

                    h_returned = True

                elif interval_thresh[channel]:
                    t_micros[channel].append(t)
                    thresholds[channel].append(interval_thresh[channel][-1])
                else:
                    print('Skipping timestamp with no defined threshold:', line)

            elif m := REGEX_HEARTBEAT.match(line):
                # see if the last heartbeat has been detected
                if not h_returned:
                    print('Heartbeat at', h_rpi[-1], 'not returned. Removing...')
                    h_rpi = h_rpi[:-1]

                date_str = m.group(1)
                date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')

                timestamp = date.timestamp()

                if args.tmin and timestamp < args.tmin: continue
                if args.tmax and timestamp > args.tmax: break

                h_rpi.append(timestamp)
                h_returned = False

                # set this as the beginning of an unbiased interval
                # if the threshold has changed
                if new_thresh:
                    interval_ti.append(timestamp * 1000)
                    new_thresh = False

                last_heartbeat = timestamp * 1000

            elif m := REGEX_THRESHOLD.match(line):
                thresh_arr = np.array(m.group(1,2,3), dtype=int)
                interval_thresh['A'].append(thresh_arr[0])
                interval_thresh['B'].append(thresh_arr[1])
                interval_thresh['C'].append(thresh_arr[2])

                new_thresh = True
                if last_heartbeat:
                    interval_tf.append(last_heartbeat)
            
            elif line.startswith('Input: '): continue 
            elif line.startswith('Shutting down: '):
                if not h_returned:
                    # remove last heartbeat
                    h_rpi.pop(-1)
                break
            
            else:
                print(line)
 
        for ch in CHANNELS:
            millis_interp = np.interp(t_micros[ch], h_ard, h_rpi)
            millis_linear = (np.array(t_micros[ch]) - h_ard[0])/1e6 + h_rpi[0]
        
            timestamps[ch] += t_micros[ch]
            millis_rpi[ch] += list((1000*np.where(millis_interp > h_rpi[0], 
                    millis_interp, 
                    millis_linear)).astype(int))

        # fast forward the time base for the next file

        # ok if this isn't exact, because linear interpolation will
        # correct the timestamps
        t_base += 2**32

        interval_tf.append(last_heartbeat)
        last_heartbeat = 0

    
    print('Done! Saving results to', args.out)
    print('Heartbeats sent:     ', len(h_rpi))
    print('Heartbeats returned: ', len(h_ard))    

    if len(h_rpi) != len(h_ard):
        print('Error: missing arduino timestamps')
        quit()


    millis_rpi = {ch: np.array(millis) for ch, millis in millis_rpi.items()}
    nonempty = {ch: millis.size > 0 for ch, millis in millis_rpi.items()}

    # use heartbeats during active DAQ as limits
    t0 = min(millis_rpi[ch].min() for ch in CHANNELS if nonempty[ch])
    t1 = max(millis_rpi[ch].max() for ch in CHANNELS if nonempty[ch])

    h_rpi = np.array(h_rpi) * 1000
    ti = h_rpi[h_rpi > t0].min()
    tf = h_rpi[h_rpi < t1].max() 

    cut_a = (millis_rpi['A'] > ti) & (millis_rpi['A'] < tf)
    cut_b = (millis_rpi['B'] > ti) & (millis_rpi['B'] < tf)
    cut_c = (millis_rpi['C'] > ti) & (millis_rpi['C'] < tf)

    # now output the results
    np.savez(args.out,
            micros_a = np.array(timestamps['A'])[cut_a],
            micros_b = np.array(timestamps['B'])[cut_b],
            micros_c = np.array(timestamps['C'])[cut_c],
            thresh_a = np.array(thresholds['A'])[cut_a],
            thresh_b = np.array(thresholds['B'])[cut_b],
            thresh_c = np.array(thresholds['C'])[cut_c],
            millis_a = np.array(millis_rpi['A'])[cut_a],
            millis_b = np.array(millis_rpi['B'])[cut_b],
            millis_c = np.array(millis_rpi['C'])[cut_c],
            interval_ti = interval_ti,
            interval_tf = interval_tf,
            interval_a = interval_thresh['A'],
            interval_b = interval_thresh['B'],
            interval_c = interval_thresh['C'],
            )
