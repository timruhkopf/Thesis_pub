#!/bin/bash
# requiries



# NOTICE Execution via bash shell/Jobs/Unittests.sh
# install packages:
module load python/3.8.2  # remember to load this before calling python3!
# python3 -m  pip install torch
# python3 -m pip install git+https://github.com/AdamCobb/hamiltorch


echo "#### conquering the snake ####"
echo 'currently at dir: ' $PWD
echo 'be aware to change to /Thesis/ and start script using "bash shell/Jobs/Unittests.sh"'
#cd /home/tim/PycharmProjects/Thesis/
pip3 freeze > requirements.txt

# make sure not to add .py ending when calling a module file
python3 -m Pytorch.Experiments.Hidden_Nuts  &>/usr/users/truhkop/Thesis/Hidden_Nuts_run.out
python3 -m Pytorch.Experiments.GAM_Nuts &>/usr/users/truhkop/Thesis/GAM_Nuts_run.out
# python /home/uni08/truhkop/Masterthesis/Python/test_server_main.py &>/home/uni08/truhkop/results &

echo "#### finished successfully ####"
wait
