import torch.nn as nn
from Pytorch.Grid.Grid_GAM_Cases import GRID_Layout_STRUCTURED
from Pytorch.Models.StructuredBNN import StructuredBNN

from Pytorch.Grid.Suits.Suit_Samplers import samplers
import numpy as np

# (CONFIG) ---------------------------------------------------------------------
cls = StructuredBNN
cls_Grid = GRID_Layout_STRUCTURED

steps = 1000
n = 1000
n_val = 100
batch = 100

# # ALPHA CDF ------------------------------------------
# model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
#                    final_activation=nn.Identity(), shrinkage='ghorse',
#                    no_basis=20, seperated=False, bijected=True, alpha_type='constant')
#
# samplers(cls, cls_Grid, n, n_val, model_param, steps, batch, epsilons=np.arange(0.0001, 0.02, 0.002),
#          Ls=[1, 2, 3], repeated=15, name='constF')

# ALPHA CONSTANT -------------------------------------
# model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
#                    final_activation=nn.Identity(), shrinkage='ghorse',
#                    no_basis=20, seperated=False, bijected=True, alpha_type='constant')
#
# samplers(cls, cls_Grid, n, n_val, model_param, steps, batch,
#          epsilons=np.arange(0.0001, 0.02, 0.002),
#          Ls=[1, 2, 3], repeated=15)

import os

git = 'a83b999cdfF'  # hash for folder to continue  a specific folder
# git = check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip(),
# base = '/'.join(os.path.abspath(__file__).split('/')[:-3])  # for local machine
base = '/usr/users/truhkop/Thesis/Pytorch'
rooting = base + '/Experiment/Result_{}'.format(git)
#
# rooting = '/usr/users/truhkop/Thesis/Pytorch/Experiment/Result_0365244'  # this on server

grid = cls_Grid(root=rooting)

m = grid.find_successfull(path=rooting,
                          model=cls.__name__)
m = dict()
for folder in os.listdir(rooting)[:-1]:
    model, sampler = folder.split('_')
    m[folder] = list()
    for file in os.listdir(rooting + '/' + folder):
        if file.endswith('.model'):
            m[folder].append(file.split('.')[0])

grid.continue_sampling_successfull(
    n=10000, n_val=100, n_samples=10000, burn_in=10000, models=m)
