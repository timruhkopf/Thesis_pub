import os
import torch.nn as nn
from Pytorch.Experiments.Grid_BNN import BNN_Grid
from Pytorch.Models.BNN import BNN
from Pytorch.Experiments.Suit_Samplers import samplers

cls = BNN
name = cls.__name__
cls_Grid = BNN_Grid

steps = 10000 * 2
n = 1000
n_val = 100
batch = 100

rooting = lambda run: os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
    os.getcwd() + '/Results/{}/'.format(run)

model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
                   final_activation=nn.Identity(),
                   prior='normal',
                   heteroscedast=False)

rooting = lambda run: os.getcwd() + '/Results/{}/'.format(run) \
    if os.path.isdir(os.getcwd()) else os.getcwd() + '/Results/{}/'.format(run)

samplers(name, cls, cls_Grid, n, n_val, model_param, steps, batch)
