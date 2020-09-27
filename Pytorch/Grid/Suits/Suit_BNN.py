
import torch.nn as nn
from Pytorch.Grid.Grid_Layout import GRID_Layout
from Pytorch.Models.BNN import BNN
from Pytorch.Grid.Suits.Suit_Samplers import samplers
import numpy as np

cls = BNN
cls_Grid = GRID_Layout

steps = 1000
n = 1000
n_val = 100
batch = 100


model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
                   final_activation=nn.Identity(),
                   prior='normal',
                   heteroscedast=False)
samplers(cls, cls_Grid, n, n_val, model_param, steps, batch,
         epsilons=np.arange(0.0001, 0.02, 0.002), Ls=[1, 2, 3], repeated=10)
