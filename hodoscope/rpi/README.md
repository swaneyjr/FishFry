# Configuring the RPi
## Installation

To install the logging scripts on the RPi, the following python packages must first be installed:

```
$ sudo pip3 install pyserial multitimer setproctitle psutil
```

Then clone this repository onto the RPi and execute the install script:

```
$ git clone https://git.crayfis.io/crayfis/FishFry.git
$ ./FishFry/hodoscope/remote/install.sh
```

This will move the appropriate files to `$HOME/hodoscope` and install a crontab which manages the logserial daemon.

## Running the logserial daemon

A configuration file will be copied to the work directory at `$HOME/hodoscope/config.txt`. To start the daemon, simply change the first line to:

```
active True
```

When the cronjob executes in the next minute, a process named `logserial` should be visible when executing `top`, and logfiles will be written to `$HOME/hodoscope/logs/` Various features of the logserial daemon can also be modified in this way; in particular, the daemon can be run in the foreground by setting `debug` to `True` and running:

```
$ hodoscope/logserial.sh
```

### Syncing with craydata

To sync the log files, `fishstand-data@craydata.ps.uci.edu` must first have the RPi listed as an authorized user. Once the RPi is whitelisted, the second line in the crontab can be un-commented, which rsyncs the `logs` directory to `/data/hodoscope` on craydata.
