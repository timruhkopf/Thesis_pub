import torch.nn as nn
from Pytorch.Grid.Grid_Layout import GRID_Layout
from Pytorch.Models.BNN import BNN
from Pytorch.Grid.Util.Suit_Samplers import samplers
import numpy as np
import os
from subprocess import check_output
from copy import deepcopy

cls = BNN
cls_Grid = GRID_Layout

steps = 1000
n = 1000
n_val = 100
batch = 100

model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
                   final_activation=nn.Identity(),
                   prior='normal',
                   heteroscedast=False)

epsilons = np.arange(0.0001, 0.02, 0.002)
Ls = [1, 2, 3]
repeated = 15

git = check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
samp_name = cls.__name__

base = '/'.join(os.path.abspath(__file__).split('/')[:-3])
rooting = lambda run: base + '/Experiment/Result_{}/{}/'.format(git, run)

path = base + '/Experiment/'
if not os.path.isdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        print('file result existed, still continuing')

path = base + '/Experiment/Result_{}/'.format(git)
if not os.path.isdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        print('file result existed, still continuing')

run = samp_name + '_SGRLD'
root = rooting(run)

bnn_unittest = cls_Grid(root)
prelim_configs = bnn_unittest.grid_exec_SGRLD(steps=steps, batch_size=batch, epsilons=epsilons)

for config in prelim_configs:
    for i in range(repeated):
        config_copy = {key: deepcopy(value) for key, value in config.items()}
        bnn_unittest.main(
            seperated=False,
            n=n, n_val=n_val,
            model_class=cls, model_param=model_param,
            sampler_name='SGRLD', sampler_param=config_copy)

# (Continuation) ----------------------
import os

print(git)
# git = '0365244'  # hash for folder to continue
base = '/'.join(os.path.abspath(__file__).split('/')[:-3])
rooting = base + '/Experiment/Result_{}'.format(git)
# base = '/usr/users/truhkop/Thesis/Pytorch'
# rooting = base + '/Experiment/Result_{}'.format(git)

grid = cls_Grid(root=rooting)  # FIXME: SPLIT ROOTING AND CONTINUE
m = grid.find_successfull(path=rooting,
                          model=cls.__name__)
# m.pop('BNN_SGRHMC')
m = dict(BNN_SGRLD=m['BNN_SGRLD'])

grid.continue_sampling_successfull(
    n=10000, n_val=100, n_samples=10000, burn_in=10000, models=m)
