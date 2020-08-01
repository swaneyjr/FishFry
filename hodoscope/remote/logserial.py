#!/usr/bin/env python3

import sys
import os
from datetime import datetime
import threading
from time import sleep

import psutil
from setproctitle import setproctitle
from serial import Serial
from multitimer import MultiTimer

try:
    import RPi.GPIO as GPIO
except RuntimeError:
    print("Error importing RPi.GPIO!  This is probably because you need superuser privileges.")
    exit(1)

# custom process name for easier lookup
PROC_TITLE = 'logseriald' 
LOCK = threading.Lock()

def synclog(*message):
    with LOCK:
        print(*message)
    

# auxiliary class for parsing config files
class LogConfig():

    DEFAULT_CFG = {
            'active': False,
            'thresh_a': 255,
            'thresh_b': 255,
            'thresh_c': 255,
            'update_period': 600,
            'heartbeat_period': 10,
            'heartbeat_pin': 32,
            'serial_port': "/dev/ttyACM0",
            'baud_rate': 115200, 
            }
    CONFIG_TYPES = {
            'active': bool,
            'thresh_a': int,
            'thresh_b': int,
            'thresh_c': int,
            'update_period': int,
            'heartbeat_period': int,
            'heartbeat_pin': int,
            'serial_port': str,
            'baud_rate': int,
            }

    def __init__(self, cfg_file):
        self.filename = cfg_file 

        cfg = LogConfig.DEFAULT_CFG.copy()

        # parse k-v pairs
        try:
            f = open(cfg_file, 'r')
            
            # typecast valid entries 
            for l in f.readlines():
                k, v = l.split(None, 1)
                if not k in LogConfig.CONFIG_TYPES:
                    continue
                if LogConfig.CONFIG_TYPES[k] == bool:
                    v = (v.lower().strip() == 'true')
                else:
                    try:
                        v = LogConfig.CONFIG_TYPES[k](v.strip())
                    except:
                        synclog('Could not parse line', l)
                        pass

                cfg[k] = v

            f.close()

        except FileNotFoundError:
            # autogenerate default config.txt
            f = open(cfg_file, 'w')
            for kdef, vdef in cfg.items():
                f.write(kdef, str(vdef))

            f.close()
            return

        self.last_modified = os.path.getmtime(self.filename)

        # this allows elements to be accessed as attributes
        self.__dict__.update(cfg)



    def update(self):
        if os.path.getmtime(self.filename) > self.last_modified:
            self.__init__(self.filename)
            return True
        return False


class LogMasterDaemon(threading.Thread):

    def __init__(self, config, logdir):
        super().__init__(daemon=False)
        self.cfg = config
        self.logdir = logdir
        
        self.logserial_daemon = LogSerialDaemon(self.cfg)
        self.heartbeat_daemon = HeartbeatDaemon(self.cfg)

    def run(self): 

        if self.logdir is None:
            print("running in foreground.")
        else:
            now = datetime.now()
            logfile = "log_" + now.strftime("%Y%m%d-%H%M%S") + ".txt"
            print("moving process to background, follow at:", logfile)

            # use os.fork() to daemonize the process
            pid = os.fork()
            if pid != 0:
                os._exit(0)
                
            os.setsid()    
            # fork a second child to prevent zombies
            pid = os.fork()
            if pid != 0:
                os._exit(0)

            log = open(os.path.join(self.logdir, logfile), 'w')
            sys.stdout = log
            sys.stderr = log
            synclog("new log started at", now)

        self.logserial_daemon.start()
        self.heartbeat_daemon.start()

        # main loop
        while True:
            if self.cfg.update():
                if not self.cfg.active: break
                self.logserial_daemon.is_restarting = True
                self.heartbeat_daemon.interval = self.cfg.heartbeat_period

            sleep(self.cfg.update_period)

        # wait for daemons to exit
        ls_alive = True
        hb_alive = True
        count = 0
        while (ls_alive or hb_alive) and count < 5:
            if ls_alive:
                ls_alive = self.logserial_daemon.is_alive()
            if hb_alive:
                hb_alive = not self.heartbeat_daemon.stopped

            sleep(self.cfg.heartbeat_period)

            count += 1

        if ls_alive:
            synclog('Timed out waiting for LogSerialDaemon')
        if hb_alive:
            synclog('Timed out waiting for HeartbeatDaemon')

        synclog('Master thread terminating')


    @staticmethod
    def is_alive():
        for proc in psutil.process_iter():
            if proc.name() == PROC_TITLE:
                status = proc.status()
                if status != psutil.STATUS_DEAD and status != psutil.STATUS_ZOMBIE:
                    return True
        
        return False


class LogSerialDaemon(threading.Thread):
 
    def __init__(self, config):
        super().__init__(daemon=True)
        self.cfg = config 

    def run(self):

        # open the serial connection (this starts the arduino)
        ser = Serial(self.cfg.serial_port, self.cfg.baud_rate, timeout=None)
        sleep(3)

        synclog(ser.readline().decode('utf-8').strip()) 
        
        self.post_thresholds(ser)

        sys.stdout.flush()

        while True:
            self.is_restarting = False
            while not self.is_restarting:
                synclog(ser.readline().decode('utf-8').strip())
                sys.stdout.flush()

                if not self.cfg.active: 
                    ser.close()
                    return

            # updated config file
            if ser.port != self.cfg.serial_port or ser.baudrate != self.cfg.baud_rate:
                # serial port settings have changed, so reconnect
                ser.close()
                self.run()

            else:
                self.post_thresholds(ser)


    def post_thresholds(self, ser):
        s = 'A: {}\n'.format(self.cfg.thresh_a) \
          + 'B: {}\n'.format(self.cfg.thresh_b) \
          + 'C: {}\n'.format(self.cfg.thresh_c)
        ser.write(s.encode('utf-8'))
        ser.flushInput()


class HeartbeatDaemon(MultiTimer):
    def __init__(self, config):
        super().__init__(config.heartbeat_period, 
                HeartbeatDaemon.heartbeat, 
                kwargs={'self': self}) 
        self.cfg = config
        self.stopped = True
        
    def start(self):
        if not self.stopped: return
        self.stopped = False

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.cfg.heartbeat_pin, GPIO.OUT)
        GPIO.output(self.cfg.heartbeat_pin, GPIO.HIGH)
        super().start()
        self._timer._daemonic = True
    
    def heartbeat(self):
        
        GPIO.output(self.cfg.heartbeat_pin, GPIO.LOW)
        synclog("heartbeat:", datetime.now())
        sleep(0.01)
        GPIO.output(self.cfg.heartbeat_pin, GPIO.HIGH) 
        sys.stdout.flush()

        if not self.cfg.active: 
            self.stop()

    def stop(self):
        super().stop()
        GPIO.cleanup()
        self.stopped = True


if __name__ == "__main__": 

    if LogMasterDaemon.is_alive():
        # nothing to do
        print('Process is still alive.  Exiting')
        exit(0)

    setproctitle(PROC_TITLE)

    from argparse import ArgumentParser

    parser = ArgumentParser('Write arduino triggers to logs')
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--log', help='directory to store log files')
    parser.add_argument('--debug', action='store_true', help='run in foreground')

    args = parser.parse_args() 

    cfg = LogConfig(args.config)
    if not cfg.active:
        exit(0)

    print("starting new logging thread.")
    log = None if args.debug else args.log
    t = LogMasterDaemon(cfg, log)
    t.start()



