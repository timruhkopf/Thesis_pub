import numpy as np
import torch.nn as nn
from Pytorch.Layer.Group_HorseShoe import Group_HorseShoe
from Pytorch.Grid.Grid_Layout import GRID_Layout
from Pytorch.Grid.Suits.Suit_Samplers import samplers

# (CONFIG) ---------------------------------------------------------------------
steps = 1000
n = 1000
n_val = 100
batch = 100

# (SAMPLER CHECK UP) -----------------------------------------------------------
cls = Group_HorseShoe
cls_Grid = GRID_Layout

model_param = dict(no_in=2, no_out=1, bias=True, activation=nn.ReLU(), bijected=True)

samplers(cls, cls_Grid, n, n_val, model_param, steps, batch,
         epsilons=np.arange(0.0001, 0.02, 0.002),
         Ls=[1, 2, 3], repeated=30)

# (Continuation) ----------------------
import os

git = '0365244'  # hash for folder to continue
base = '/'.join(os.path.abspath(__file__).split('/')[:-3])
rooting = base + '/Experiment/Result_{}'.format(git)
# rooting = '/usr/users/truhkop/Thesis/Pytorch/Experiment/Result_0365244'

grid = cls_Grid(root=rooting)
m = grid.find_successfull(path=rooting,
                          model=cls.__name__)
m = {'Grouped_HorseShoe_MALA': m['Grouped_HorseShoe_MALA'],
     'Grouped_HorseShoe_SGLD': m['Grouped_HorseShoe_SGLD'],
     'Grouped_HorseShoe_SGNHT': m['Grouped_HorseShoe_SGNHT']}
print(m)
grid.continue_sampling_successfull(
    n=1000, n_val=100, n_samples=1000, burn_in=1000, models=m)
