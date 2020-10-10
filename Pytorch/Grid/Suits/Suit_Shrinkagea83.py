import os
import torch.nn as nn
from Pytorch.Grid.Grid_Layout import GRID_Layout
from Pytorch.Models.ShrinkageBNN import ShrinkageBNN

from Pytorch.Grid.Util.Suit_Samplers import samplers
import numpy as np

# (CONFIG) ---------------------------------------------------------------------

cls = ShrinkageBNN
cls_Grid = GRID_Layout

steps = 1000
n = 1000
n_val = 100
batch = 100

# (seperated ) ----------------------------------------------------------
# model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
#                    final_activation=nn.Identity(),
#                    shrinkage='ghorse', seperated=False, bijected=True,
#                    heteroscedast=False)
#
# samplers(cls, cls_Grid, n, n_val, model_param, steps, batch,
#          epsilons=np.arange(0.0001, 0.02, 0.002),
#          Ls=[1, 2, 3], repeated=15)

# (seperated = False) ----------------------------------------------------------
# model_param = dict(hunits=[2, 10, 5, 1], activation=nn.ReLU(),
#                    final_activation=nn.Identity(),
#                    shrinkage='ghorse', seperated=False, bijected=True,
#                    heteroscedast=False)
#
# samplers(cls, cls_Grid, n, n_val, model_param, steps, batch,
#          epsilons=np.arange(0.0001, 0.02, 0.002),
#          Ls=[1, 2, 3], repeated=15)
#

# (Continuation) ----------------------
import os

git = 'a83b999'  # hash for folder to continue  a specific folder
# git = check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip(),
# base = '/'.join(os.path.abspath(__file__).split('/')[:-3])  # for local machine
base = '/usr/users/truhkop/Thesis/Pytorch'  # for server
rooting = base + '/Experiment/Result_{}'.format(git)

grid = cls_Grid(root=rooting)
m = grid.find_successfull(path=rooting,
                          model=cls.__name__, n=2)
# m = {'ShrinkageBNN_SGRHMC':['ShrinkageBNN_SGRHMC_a83b999_095925603259']}
grid.continue_sampling_successfull(
    n=10000, n_val=100, n_samples=10000, burn_in=10000, models=m)
