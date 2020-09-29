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

git = '0365244'  # hash for folder to continue
base = '/'.join(os.path.abspath(__file__).split('/')[:-3])
rooting = base + '/Experiment/Result_{}'.format(git)

grid = cls_Grid(root=rooting)
m = grid.find_successfull(path='/home/tim/PycharmProjects/Thesis/Pytorch/Experiment/Result_0365244',
                          model=cls.__name__)
grid.continue_sampling_successfull(
    n=1000, n_val=100, n_samples=10000, burn_in=10000, models=m)
