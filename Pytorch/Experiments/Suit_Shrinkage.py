import os
import torch.nn as nn
from Pytorch.Experiments.Grid_BNN import BNN_Grid
from Pytorch.Models.ShrinkageBNN import ShrinkageBNN

from Pytorch.Experiments.Suit_Samplers import samplers
# (CONFIG) ---------------------------------------------------------------------

cls = ShrinkageBNN
cls_Grid = BNN_Grid
name = cls.__name__

steps = 10000 * 2
n = 10000
n_val = 100
batch = 100

model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
                   final_activation=nn.Identity(),
                   shrinkage='glasso', seperated=True, bijected=True,
                   heteroscedast=False)

rooting = lambda run: os.getcwd() + '/Results/{}/'.format(run) \
    if os.path.isdir(os.getcwd()) else os.getcwd() + '/Results/{}/'.format(run)

samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch)
