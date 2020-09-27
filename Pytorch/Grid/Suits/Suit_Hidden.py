import numpy as np
import torch.nn as nn
from Pytorch.Layer.Hidden import Hidden
from Pytorch.Experiments.Grid_Layer import Layer_Grid
from Pytorch.Suits.SUIT_Samplers import samplers

# (CONFIG) ---------------------------------------------------------------------
steps = 1000
n = 1000
n_val = 100
batch = 100

# (SAMPLER CHECK UP) -----------------------------------------------------------
# USING Hidden

run = 'Unittest_samplers'

cls = Hidden
name = cls.__name__
cls_Grid = Layer_Grid

model_param = dict(no_in=2, no_out=1, bias=True, activation=nn.ReLU())

samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch,
         epsilons=np.arange(0.0001, 0.05, 0.003),
         Ls=[1, 2, 3, 5])

samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch,
         epsilons=np.arange(0.0001, 0.05, 0.003),
         Ls=[1, 2, 3, 5])

samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch,
         epsilons=np.arange(0.0001, 0.05, 0.003),
         Ls=[1, 2, 3, 5])
