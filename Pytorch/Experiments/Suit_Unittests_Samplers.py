import os
import torch.nn as nn
from Pytorch.Layer.Hidden import Hidden, Hidden_flat
from Pytorch.Experiments.Grid_Layer import Layer_Grid

# (CONFIG) ---------------------------------------------------------------------
steps = 1000
n = 1000
n_val = 100
batch = 100
rooting = lambda run: os.getcwd() + '/Results/{}/'.format(run) if os.path.isdir(os.getcwd()) else \
    os.getcwd() + '/Results/{}/'.format(run)

# (SAMPLER CHECK UP) -----------------------------------------------------------
# USING Hidden

run = 'Unittest_samplers'
root = rooting(run)

cls = Hidden
name = cls.__name__
cls_Grid = Layer_Grid

model_param = dict(no_in=2, no_out=1, bias=True, activation=nn.ReLU())

# (0) SGNHT --------------------------------------------------------------------
run = name + '_SGNHT'
root = rooting(run)

bnn_unittest = cls_Grid(root)
prelim_configs = bnn_unittest.grid_exec_SGNHT(steps=steps, batch_size=batch)

for config in prelim_configs:
    bnn_unittest.main(
        n=n, n_val=n_val, seperated=True,
        model_class=cls, model_param=model_param,
        sampler_name='SGNHT', sampler_param=config)

# (1) MALA -------------------------------------------------
run = name + '_MALA'
root = rooting(run)

bnn_unittest = cls_Grid(root)
prelim_configs = bnn_unittest.grid_exec_MALA(steps, batch_size=batch)

for config in prelim_configs:
    bnn_unittest.main(
        n=n, n_val=n_val, seperated=True,
        model_class=cls, model_param=model_param,
        sampler_name='MALA', sampler_param=config)

# (2) SGLD -------------------------------------------------
run = name + '_SGLD'
root = rooting(run)

bnn_unittest = cls_Grid(root)
prelim_configs = bnn_unittest.grid_exec_SGLD(steps, batch_size=batch)

for config in prelim_configs:
    bnn_unittest.main(
        n=n, n_val=n_val, seperated=True,
        model_class=cls, model_param=model_param,
        sampler_name='SGLD', sampler_param=config)

# (3) RHMC -------------------------------------------------

run = name + '_RHMC'
root = rooting(run)
bnn_unittest = cls_Grid(root)
prelim_configs = bnn_unittest.grid_exec_RHMC(steps=steps)

for config in prelim_configs:
    bnn_unittest.main(
        n=n, n_val=n_val, seperated=True,
        model_class=cls, model_param=model_param,
        sampler_name='RHMC', sampler_param=config)

# (4) SGRLD -------------------------------------------------
run = name + '_SGRLD'
root = rooting(run)

bnn_unittest = cls_Grid(root)
prelim_configs = bnn_unittest.grid_exec_SGRLD(steps=steps, batch_size=100)

for config in prelim_configs:
    bnn_unittest.main(
        n=n, n_val=n_val, seperated=True,
        model_class=cls, model_param=model_param,
        sampler_name='SGRLD', sampler_param=config)

# (4) SGRHMC -------------------------------------------------
run = name + '_SGRHMC'
root = rooting(run)

bnn_unittest = cls_Grid(root)
prelim_configs = bnn_unittest.grid_exec_SGRHMC(steps, batch)

for config in prelim_configs:
    bnn_unittest.main(
        n=n, n_val=n_val, seperated=True,
        model_class=cls, model_param=model_param,
        sampler_name='SGRHMC', sampler_param=config)
