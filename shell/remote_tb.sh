#!/bin/bash
# Script to run Tensorboard from remote machine & forward it to local machine

# Open Tensorboard from input
module purge
module load python/3.6.3
source $HOME/MA-3.6.3/bin/activate # activate venv

# ToDo get input for partial dir
#echo 'current wd:'
cd $HOME/results/
# printing dir with no files
# ls -R | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\// /g' -e 's/^/ /'
find . | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"

echo 'please refresh browser'
# since ssh opens 6006,
fuser -k 6007/tcp
tensorboard --logdir=train --port=6007



# CLEAN UP OF PORT
netstat -tulpn | grep 127.0.0
lsof | grep ruhkop | grep TCP
netstat -ntap

# kill port 6007
fuser -k 6007/tcp

