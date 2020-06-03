#!/bin/bash

DIR=$(dirname $0)

WORK_DIR=$HOME/hodoscope/

mkdir -p $WORK_DIR
mkdir -p $WORK_DIR/logs

cp $DIR/*.py $DIR/*.txt $DIR/logserial.sh $WORK_DIR
cd $WORK_DIR

crontab cron.txt
