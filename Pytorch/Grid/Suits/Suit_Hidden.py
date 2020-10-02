import numpy as np
import torch.nn as nn
from Pytorch.Layer.Hidden import Hidden
from Pytorch.Grid.Grid_Layout import GRID_Layout
from Pytorch.Grid.Suits.Suit_Samplers import samplers

# (CONFIG) ---------------------------------------------------------------------
steps = 1000
n = 1000
n_val = 100
batch = 100

# (SAMPLER CHECK UP) -----------------------------------------------------------
cls = Hidden
cls_Grid = GRID_Layout
#
# model_param = dict(no_in=2, no_out=1, bias=True, activation=nn.ReLU())
#
# samplers(cls, cls_Grid, n, n_val, model_param, steps, batch,
#          epsilons=np.arange(0.0001, 0.02, 0.002),
#          Ls=[1, 2, 3], repeated=30)

# (Continuation) ----------------------
import os

git = '390663d9'  # hash for folder to continue
# base = '/'.join(os.path.abspath(__file__).split('/')[:-3])
# rooting = base + '/Experiment/Result_{}'.format(git)
rooting = '/usr/users/truhkop/Thesis/Pytorch/Experiment/Result_{}'.format(git)

grid = cls_Grid(root=rooting)
m = grid.find_successfull(path=rooting,
                          model=cls.__name__)
# m = {'Hidden_SGNHT': m['Hidden_SGNHT'],
#      'Hidden_MALA': m['Hidden_MALA'],
#      'Hidden_SGLD': m['Hidden_SGLD']}

grid.continue_sampling_successfull(
    n=1000, n_val=100, n_samples=10000, burn_in=10000, models=m)
