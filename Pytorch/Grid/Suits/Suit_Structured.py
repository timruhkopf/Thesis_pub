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

# ALPHA CDF ------------------------------------------
model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
                   final_activation=nn.Identity(), shrinkage='ghorse',
                   no_basis=20, seperated=True, bijected=True, alpha_type='cdf')

samplers(cls, cls_Grid, n, n_val, model_param, steps, batch, epsilons=np.arange(0.0001, 0.02, 0.002),
         Ls=[1, 2, 3], repeated=15)

# ALPHA CONSTANT -------------------------------------
model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
                   final_activation=nn.Identity(), shrinkage='ghorse',
                   no_basis=20, seperated=True, bijected=True, alpha_type='constant')

samplers(cls, cls_Grid, n, n_val, model_param, steps, batch,
         epsilons=np.arange(0.0001, 0.02, 0.002),
         Ls=[1, 2, 3], repeated=15)
