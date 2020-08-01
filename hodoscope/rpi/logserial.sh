#/bin/bash
WORKDIR=$(dirname $0)
$WORKDIR/logserial.py --config $WORKDIR/config.txt --log $WORKDIR/logs/
