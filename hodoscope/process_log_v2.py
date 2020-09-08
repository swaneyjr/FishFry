#!/usr/bin/env python3

import re
from datetime import datetime

import numpy as np

REGEX_TIMESTAMP = re.compile(r'([ABCH]) (\d+)')
REGEX_HEARTBEAT = re.compile(r'heartbeat:\s+(\S+\s+\S+)') 
REGEX_THRESHOLD = re.compile(r'pwm thresholds: (\d+) (\d+) (\d+)')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with open(args.infile, 'r') as f_in:

        current_thresh = {'A': 255, 'B': 255, 'C': 255}

        thresholds = {c: [] for c in ('A','B','C')}
        timestamps = {c: [] for c in ('A','B','C')}

        # handle wraparound in microsecond timestamps after 2^32
        last_t = 0
        t_base = 0

        # synchronize time bases of heartbeats
        h_returned = True
        h_ard = []
        h_rpi = []

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

                    h_returned = True
                else:
                    timestamps[channel].append(t)
                    thresholds[channel].append(current_thresh[channel])

            elif m := REGEX_HEARTBEAT.match(line):
                # see if the last heartbeat has been detected
                if not h_returned:
                    print('Heartbeat at', h_rpi[-1], 'not returned. Removing...')
                    h_rpi = h_rpi[:-1]

                date_str = m.group(1)
                date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')

                h_rpi.append(date.timestamp())
                h_returned = False

            elif m := REGEX_THRESHOLD.match(line):
                thresh_arr = np.array(m.group(1,2,3), dtype=int)
                current_thresh['A'] = thresh_arr[0]
                current_thresh['B'] = thresh_arr[1]
                current_thresh['C'] = thresh_arr[2]
            
            elif line.startswith('Input: '): continue 
            elif line.startswith('Shutting down: '):
                if not h_returned:
                    # remove last heartbeat
                    h_rpi.pop(-1)
                break
            
            else:
                print(line)
    
    print('Done! Saving results to', args.out)
    print('Heartbeats sent:     ', len(h_rpi))
    print('Heartbeats returned: ', len(h_ard))

    if len(h_rpi) != len(h_ard):
        print('Error: missing arduino timestamps')
        quit()

    millis_rpi = {}
    for ch in ('A','B','C'):
        millis_interp = np.interp(timestamps[ch], h_ard, h_rpi)
        millis_linear = (np.array(timestamps[ch]) - h_ard[0])/1e6 + h_rpi[0]
        
        millis_rpi[ch] = (1000*np.where(millis_interp > h_rpi[0], 
                millis_interp, 
                millis_linear)).astype(int)
        

    # now output the results
    np.savez(args.out,
            micros_a = timestamps['A'],
            micros_b = timestamps['B'],
            micros_c = timestamps['C'],
            thresh_a = thresholds['A'],
            thresh_b = thresholds['B'],
            thresh_c = thresholds['C'],
            millis_a = millis_rpi['A'],
            millis_b = millis_rpi['B'],
            millis_c = millis_rpi['C'],
            )
