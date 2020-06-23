#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 1
#SBATCH --gres=gpu:1
#SBATCH --job-name=test_thesis
#SBATCH --output=test_thesis.%j.out

# gpu selection
# https://info.gwdg.de/dokuwiki/doku.php?id=en:services:application_services:high_performance_computing:running_jobs_slurm#gpu_selection

module purge
module load cuda90/fft
module load cuda90/nsight
module load cuda90/profiler

# GWDDG tutorial on tensorflow
module load cuda90/toolkit/9.0.176
module load cuda90/blas/9.0.176
module load cudnn/90v7.3.1

# user defined module (e.g. python 3.8)
# https://lmod.readthedocs.io/en/latest/020_advanced.html
#module use $HOME/modulefiles # prepend the user defined modulefiles
#module load openmpi/gcc/64   # dependency
#module load openmpi/gcc      # dependency
#module load python/3.8.0     # cmd fails!


module load python/3.6.3
module load R/3.6.1

echo "#### Starting my duties ####"
# manually installed python verison:
#$HOME/build/Tensorflow-3.6.8/python -m pip freeze
#$HOME/build/Tensorflow-3.6.8/python /home/uni08/truhkop/Masterthesis/Tensorflow/test_server_main.py &> /home/uni08/truhkop/results & wait

# virtualenviroment

echo "#### conquering the snake ####"
python pip freeze
python /home/uni08/truhkop/Masterthesis/Python/test_server_main.py &> /home/uni08/truhkop/results & wait

echo "#### retaRding ####"
Rscript /home/uni08/truhkop/Masterthesis/R/tRy.R

echo "#### End of my shift ####"



