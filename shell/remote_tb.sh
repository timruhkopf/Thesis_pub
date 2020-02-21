# Script to run Tensorboard from remote machine & forward it to local machine

# ssh connection via port-forwarding
ssh -L 16006:127.0.0.1:6006 user@remote

# Open Tensorboard from input
module purge
module load python/3.6.8
source MA-3.6.3/bin/activate # activate venv

# ToDo get input for partial dir
tensorboard --logdir=results/logs

# ToDo Open Chrome with port
# FIXME: depending on which port is used!
http://127.0.0.1:16006/

# ToDo cleaning up the port
# ps aux | grep ssh | grep truhkop
# kill <id>
