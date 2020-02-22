#!/bin/bash
# Script to run Tensorboard from remote machine & forward it to local machine

# ssh connection via port-forwarding
ssh -L 16006:127.0.0.1:6006 truhkop@gwdu103.gwdg.de

# Open Tensorboard from input
module purge
module load python/3.6.3
source $HOME/MA-3.6.3/bin/activate # activate venv

# ToDo get input for partial dir
echo 'current wd:'
cd $HOME/results/logs
# printing dir with no files
# ls -R | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\// /g' -e 's/^/ /'
find . | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"

echo 'Which Tensorboard dir would you like to launch?'
read dir
tensorboard --logdir=results/logs/$dir

# Open Chrome with port
xdg-open http://127.0.0.1:16006/
read -p 'press any key to kill this port & terminate' -n1 -s


# ToDo cleaning up the port
# FIXME: depending on which port is used!
ps aux | grep ssh | grep truhkop
#kill <id>
