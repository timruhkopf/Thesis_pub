import os
import torch.nn as nn
from Pytorch.Models.GAM import GAM
from Pytorch.Experiments.Grid_GAM import GAM_Grid
from Pytorch.Experiments.Suit_Samplers import samplers

# (CONFIG) ---------------------------------------------------------------------
steps = 10000
n = 10000
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
cls_Grid = GAM_Grid

model_param = dict(xgrid=(0, 10, 0.5), order=1, no_basis=20, no_out=1,
                   activation=nn.Identity(), bijected=False, penK=True)

samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch)
