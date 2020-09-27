import torch.nn as nn
from Pytorch.Grid.Grid_GAM_Cases import GRID_Layout_STRUCTURED
from Pytorch.Models.StructuredBNN import StructuredBNN

from Pytorch.Suits.SUIT_Samplers import samplers
import numpy as np

# (CONFIG) ---------------------------------------------------------------------

cls = StructuredBNN
name = cls.__name__
cls_Grid = GRID_Layout_STRUCTURED

steps = 10000
n = 1000
n_val = 100
batch = 100

# ALPHA CDF ------------------------------------------
model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
                   final_activation=nn.Identity(), shrinkage='ghorse',
                   no_basis=20, seperated=True, bijected=True)

samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch, epsilons=np.arange(0.0001, 0.05, 0.003),
         Ls=[1, 2, 3, 5])

# ALPHA CONSTANT -------------------------------------
model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
                   final_activation=nn.Identity(), shrinkage='ghorse',
                   no_basis=20, seperated=True, bijected=True, alpha_type='constant')

samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch, epsilons=np.arange(0.0001, 0.05, 0.003),
         Ls=[1, 2, 3, 5])

samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch, epsilons=np.arange(0.0001, 0.05, 0.003),
         Ls=[1, 2, 3, 5])
