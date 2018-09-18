#!/usr/bin/env python

import argparse, datetime, threading, time, math, atexit, serial, sys, os

try:
    import RPi.GPIO as GPIO
except RuntimeError:
    print("Error importing RPi.GPIO!  This is probably because you need superuser privileges.")

PERIOD = 10    
WORK_DIR = "/home/benchtop/hodoscope/"
HALT_FILE = WORK_DIR + "halt"
HEARTBEAT_FILE = WORK_DIR + "heartbeat"
HEARTBEAT_PIN = 32

def next():
    t = threading.Timer( max(0,next_call - time.time()), heartbeat )
    t.daemon = True
    t.start()

def heartbeat():
  global next_call
  if (next_call == 0):
      next_call = 1
      return
  os.utime(HEARTBEAT_FILE, None)
  GPIO.output(HEARTBEAT_PIN, GPIO.LOW)
  print "heartbeat:  ", datetime.datetime.now()
  time.sleep(0.01)
  GPIO.output(HEARTBEAT_PIN, GPIO.HIGH)	
  next_call = next_call+PERIOD
  next()
  
def cleanup():
    global next_call
    next_call = 0
    print "waiting for heartbeat daemon to acknowledge exit...";
    count = 0;
    while(next_call == 0):
        time.sleep(1)
        count+=1
        if (count > PERIOD*1.2):
            print "timed-out waiting on heartbeat daemon."
            break
    print "cleaning up GPIO..."
    GPIO.cleanup()
    print "successful exit."

def startup(args):
    global next_call
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(HEARTBEAT_PIN, GPIO.OUT)
    GPIO.output(HEARTBEAT_PIN, GPIO.HIGH)
    
    SERIAL_PORT="/dev/ttyACM0"
    print "connecting to the Arduino..."

    # open the serial connection
    ser = serial.Serial(SERIAL_PORT, 115200,timeout=None)
    line = ser.readline().strip()
    print line

    now = datetime.datetime.now();

    print "connected at ", now;
    sys.stdout.flush()

    if (args.fg):
        print "running in foreground."
    else:
        logfile = "/home/benchtop/hodoscope/log_" + now.strftime("%Y%m%d-%H%M%S") + ".txt"
        print "moving process to background, follow at:  " + logfile

        log = open(logfile, 'w')
        sys.stdout = log
        sys.stderr = log
        newpid = os.fork()
        if newpid != 0:
            exit(0)
        print "new log started at ", datetime.datetime.now()

    next_call = PERIOD*math.ceil(time.time()/PERIOD)
    next()
    atexit.register(cleanup)
    
    while True:
        if os.path.isfile(HALT_FILE):
            exit(0)
        line = ser.readline().strip()
        print line
        sys.stdout.flush()
    
if __name__ == "__main__":
    example_text = '''examples:
    ...'''

    parser = argparse.ArgumentParser(description='Plot rate from Cosmics.', epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--halt',action="store_true", help="disable the logserial daemon.")
    parser.add_argument('--enable',action="store_true", help="enable the logserial daemon.")
    parser.add_argument('--fg',action="store_true", help="run in foreground (e.g. for debugging).")
    args = parser.parse_args()

    if (args.halt):
        open(HALT_FILE, 'wa').close()
        exit(0)

    if (args.enable):
        if os.path.isfile(HALT_FILE):
            try:
                os.remove(HALT_FILE)
            except OSError as e:  
                print ("Error: %s - %s." % (e.filename, e.strerror))
        exit(0)

    if os.path.isfile(HALT_FILE):
        print "logserial daemon is disabled."
        exit(0)
        
    if os.path.isfile(HEARTBEAT_FILE):
        t = datetime.datetime.fromtimestamp(os.path.getmtime(HEARTBEAT_FILE))
        age = (datetime.datetime.now() - t).total_seconds()
        #print "heartbeat age is:  ", age
        if (age < 2*PERIOD):
            print "recent heartbeat, doing nothing..."
            exit(0)
    else:
        open(HEARTBEAT_FILE, 'w').close()

    print "starting new logserial daemon."
    startup(args)



