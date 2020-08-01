#/bin/bash
WORKDIR=$(dirname $0)
$WORKDIR/logserial.py --config $WORKDIR/config_debug.txt --debug
