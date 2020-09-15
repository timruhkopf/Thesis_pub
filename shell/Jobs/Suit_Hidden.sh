#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 18:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=Suit_Hidden
#SBATCH --output=Suit_Hidden.%j.out
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
python3 -m Pytorch.Experiments.Suit_Hidden &>/usr/users/truhkop/Thesis/Suit_Hidden.out

wait