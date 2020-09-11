import os
from Pytorch.Layer.Hidden import Hidden_flat
from Pytorch.Layer.Group_lasso import Group_lasso
from Pytorch.Models.GAM import GAM

from Pytorch.Experiments.Grid_Layer import Layer_Grid

# (CONFIG) ---------------------------------------------------------------------
steps = 1000
n = 1000
n_val = 100
batch = 100
rooting = lambda run: os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
    os.getcwd() + '/Results/{}/'.format(run)
