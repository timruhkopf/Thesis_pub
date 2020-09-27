import os
import numpy as np
import torch.nn as nn
from Pytorch.Models.GAM import GAM
from Pytorch.Grid.Grid_GAM_Cases import GRID_Layout_GAM
from Pytorch.Suits.SUIT_Samplers import samplers

# (CONFIG) ---------------------------------------------------------------------
steps = 1000
n = 1000
n_val = 100
batch = 100
rooting = lambda run: os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
    os.getcwd() + '/Results/{}/'.format(run)

# (SAMPLER CHECK UP) -----------------------------------------------------------
# USING Hidden

run = 'Unittest_samplers'
root = rooting(run)

cls = GAM
name = cls.__name__
cls_Grid = GRID_Layout_GAM

model_param = dict(xgrid=(0, 10, 0.5), order=1, no_basis=20, no_out=1,
                   activation=nn.Identity(), bijected=False)

samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch, epsilons=np.arange(0.0001, 0.05, 0.003),
         Ls=[1, 2, 3, 5])

samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch, epsilons=np.arange(0.0001, 0.05, 0.003),
         Ls=[1, 2, 3, 5])

samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch, epsilons=np.arange(0.0001, 0.05, 0.003),
         Ls=[1, 2, 3, 5])
