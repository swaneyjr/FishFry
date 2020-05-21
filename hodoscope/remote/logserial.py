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


# auxiliary class for parsing config files
class LogConfig():

    DEFAULT_CFG = {
            'active': False,
            'debug': False,
            'update_period': 600,
            'heartbeat_period': 10,
            'heartbeat_pin': 32,
            'serial_port': "/dev/ttyACM0",
            'baud_rate': 115200,
            }
    CONFIG_TYPES = {
            'active': bool,
            'debug': bool,
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
                        print('Could not parse line', l)
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
        super().__init__(daemon=True)
        self.cfg = config
        self.logdir = logdir
        
        self.logserial_daemon = LogSerialDaemon(self.cfg)
        self.heartbeat_daemon = HeartbeatDaemon(self.cfg)

    def run(self): 

        if self.cfg.debug:
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
            print("new log started at", now)

        self.logserial_daemon.start()
        self.heartbeat_daemon.start()

        # main loop
        while True:
            if self.cfg.update():
                if not self.cfg.active: break
                self.logserial_daemon.is_restarting = True
                self.heartbeat_daemon.is_restarting = True 

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
            print('Timed out waiting for LogSerialDaemon')
        if hb_alive:
            print('Timed out waiting for HeartbeatDaemon')

        print('Master thread terminating')


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

        self.is_restarting = False

        # open the serial connection
        ser = Serial(self.cfg.serial_port, self.cfg.baud_rate, timeout=None)
        print(ser.readline().decode('utf-8').strip())
   
        sys.stdout.flush()
  
        while self.cfg.active and not self.is_restarting:
            print(ser.readline().decode('utf-8').strip())
            sys.stdout.flush()

        if self.is_restarting:
            self.run()



class HeartbeatDaemon(MultiTimer):
    def __init__(self, config):
        super().__init__(config.heartbeat_period, 
                HeartbeatDaemon.heartbeat, 
                kwargs={'self': self}) 
        self.cfg = config
        self.stopped = True
        self.is_restarting = False
        
    def start(self):
        if not self.stopped: return
        self.stopped = False
        self.is_restarting = False

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.cfg.heartbeat_pin, GPIO.OUT)
        GPIO.output(self.cfg.heartbeat_pin, GPIO.HIGH)
        super().start()
        self._timer._daemonic = True
    
    def heartbeat(self):
        
        GPIO.output(self.cfg.heartbeat_pin, GPIO.LOW)
        print("heartbeat:", datetime.now())
        sleep(0.01)
        GPIO.output(self.cfg.heartbeat_pin, GPIO.HIGH) 
        sys.stdout.flush()

        if not self.cfg.active: 
            self.stop()

        elif self.is_restarting:
            self.stop()
            self.start()

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
    parser.add_argument('--log', required=True, help='directory to store log files')

    args = parser.parse_args() 

    cfg = LogConfig(args.config)
    if not cfg.active:
        exit(0)

    print("starting new logging thread.") 
    t = LogMasterDaemon(cfg, args.log)
    t.start()



