#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=Suit_GHorse
#SBATCH --output=Suit_GHorse.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=timruhkopf@googlemail.com

# requiries
module purge
module load cuda90/fft
module load cuda90/nsight
module load cuda90/profiler

# GWDDG tutorial on tensorflow
module load cuda90/toolkit/9.0.176
module load cuda90/blas/9.0.176
module load cudnn/90v7.3.1

# NOTICE Execution via bash shell/Jobs/Unittests.sh
# install packages:
module load python/3.8.2 # remember to load this before calling python3!
# python3 -m  pip install torch
# python3 -m pip install git+https://github.com/AdamCobb/hamiltorch

echo "#### conquering the snake ####"
echo 'currently at dir: ' $PWD
echo 'be aware to change to /Thesis/ and start script using "bash shell/Jobs/Unittests.sh"'
#cd /home/tim/PycharmProjects/Thesis/

# make sure not to add .py ending when calling a module file
python3 -m Pytorch.Grid.Suits.Suit_GHorse &>/usr/users/truhkop/Thesis/Pytorch/Experiment/Suit_GHorse.out

wait
