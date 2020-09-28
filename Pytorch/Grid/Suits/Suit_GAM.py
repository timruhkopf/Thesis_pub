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

model_param = dict(xgrid=(0, 10, 0.5), order=1, no_basis=20, no_out=1,
                   activation=nn.Identity(), bijected=False)

samplers(cls, cls_Grid, n, n_val, model_param, steps, batch,
         epsilons=np.arange(0.0001, 0.02, 0.002),
         Ls=[1, 2, 3], repeated=15)
