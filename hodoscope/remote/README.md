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

A configuration file will be copied to the work directory at `$HOME/hodoscope/config.txt`.  To start the daemon, simply change the first line to:

```
active True
```

When the cronjob executes in the next minute, a process named `logserial` should be visible when executing `top`, and logfiles will be written to `$HOME/hodoscope/logs/`  Various features of the logserial daemon can also be modified in this way; in particular, the daemon can be run in the foreground by setting `debug` to `True` and running:

```
$ hodoscope/logserial.sh
```