import os
import torch.nn as nn
from Pytorch.Experiments.Grid_Structured_BNN import Structured_BNN_Grid
from Pytorch.Models.StructuredBNN import StructuredBNN

from Pytorch.Experiments.Suit_Samplers import samplers
import numpy as np

# (CONFIG) ---------------------------------------------------------------------

cls = StructuredBNN
name = cls.__name__
cls_Grid = Structured_BNN_Grid

steps = 1000
n = 1000
n_val = 100
batch = 100

# ALPHA CDF ------------------------------------------
model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
                   final_activation=nn.Identity(), shrinkage='glasso', no_basis=20,
                   seperated=True, bijected=True)

samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch)

# ALPHA CONSTANT -------------------------------------
model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
                   final_activation=nn.Identity(), shrinkage='glasso', no_basis=20,
                   seperated=True, bijected=True, alpha_type='constant')

samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch, epsilons=np.arange(0.0001, 0.05, 0.002),
         Ls=[1, 2, 3, 5])
