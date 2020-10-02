import os
import torch.nn as nn
from Pytorch.Grid.Grid_Layout import GRID_Layout
from Pytorch.Models.ShrinkageBNN import ShrinkageBNN

from Pytorch.Grid.Suits.Suit_Samplers import samplers
import numpy as np

cls = ShrinkageBNN
cls_Grid = GRID_Layout

steps = 1000
n = 1000
n_val = 100
batch = 100

git = '0365244'  # hash for folder to continue
base = '/'.join(os.path.abspath(__file__).split('/')[:-3])
rooting = base + '/Experiment/Result_{}'.format(git)

# rooting = '/usr/users/truhkop/Thesis/Pytorch/Experiment/Result_0365244'  # this on server
# '/home/tim/PycharmProjects/Thesis/Pytorch/Experiment/Result_0365244/ShrinkageBNN_RHMC'
# '/home/tim/PycharmProjects/Thesis/Pytorch/Experiment/Result_0365244/ShrinkageBNN_SGRLD'
# '/home/tim/PycharmProjects/Thesis/Pytorch/Experiment/Result_0365244/ShrinkageBNN_SGRHMC'

grid = cls_Grid(root=rooting)
m = grid.find_successfull(path=rooting,
                          model=cls.__name__, )

m.pop('ShrinkageBNN_SGRHMC')
m
grid.continue_sampling_successfull(
    n=10000, n_val=100, n_samples=10000, burn_in=10000, models=m)
