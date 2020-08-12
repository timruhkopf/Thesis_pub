#!/bin/bash
# NOTICE Execution via bash shell/Jobs/Unittests.sh
module load python/3.6.3

echo "#### conquering the snake ####"
cd /home/tim/PycharmProjects/Thesis/
# python pip freeze

# make sure not to add .py ending when calling a module file
python -m Pytorch.Experiments.Hidden_Nuts # &>/home/tim/PycharmProjects/Thesis/Pytorch/Experiments/results/
python -m Pytorch.Experiments.GAM_NUTS
# python /home/uni08/truhkop/Masterthesis/Python/test_server_main.py &>/home/uni08/truhkop/results &
wait
