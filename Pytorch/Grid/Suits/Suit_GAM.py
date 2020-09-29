import os
import numpy as np
import torch.nn as nn
from Pytorch.Layer.GAM import GAM
from Pytorch.Grid.Grid_GAM_Cases import GRID_Layout_GAM
from Pytorch.Grid.Suits.Suit_Samplers import samplers

# (CONFIG) ---------------------------------------------------------------------
steps = 1000
n = 1000
n_val = 100
batch = 100

# (SAMPLER CHECK UP) -----------------------------------------------------------
cls = GAM
cls_Grid = GRID_Layout_GAM

# model_param = dict(xgrid=(0, 10, 0.5), order=1, no_basis=20, no_out=1,
#                    activation=nn.Identity(), bijected=False)
#
# samplers(cls, cls_Grid, n, n_val, model_param, steps, batch,
#          epsilons=np.arange(0.0001, 0.02, 0.002),
#          Ls=[1, 2, 3], repeated=15)


# (Continuation) ----------------------
import os

# git = '0365244'  # hash for folder to continue
# base = '/'.join(os.path.abspath(__file__).split('/')[:-3])
# rooting = base + '/Experiment/Result_{}'.format(git)
rooting = '/usr/users/truhkop/Thesis/Pytorch/Experiment/Result_0365244'

grid = cls_Grid(root=rooting)
m = grid.find_successfull(path=rooting,
                          model=cls.__name__)
grid.continue_sampling_successfull(
    n=1000, n_val=100, n_samples=10000, burn_in=10000, models=m)
