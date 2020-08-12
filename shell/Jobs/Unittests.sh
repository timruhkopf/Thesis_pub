#!/bin/bash
# bash requiries


# NOTICE Execution via bash shell/Jobs/Unittests.sh
# install packages:
# python3 -m  pip install torch
# python3 -m pip install git+https://github.com/AdamCobb/hamiltorch
module load python/3.8.2


echo "#### conquering the snake ####"
cd /home/tim/PycharmProjects/Thesis/
pip3 freeze > requirements.txt

# make sure not to add .py ending when calling a module file
python3 -m Pytorch.Experiments.Hidden_Nuts # &>/home/tim/PycharmProjects/Thesis/Pytorch/Experiments/results/
python3 -m Pytorch.Experiments.GAM_NUTS
# python /home/uni08/truhkop/Masterthesis/Python/test_server_main.py &>/home/uni08/truhkop/results &
wait
