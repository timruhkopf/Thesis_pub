import torch
import torch.distributions as td

import Pytorch.Models as M

no_in = 2
no_units = 1
n = 100
X_dist = td.Uniform(-10. * torch.ones(no_in), 10 * torch.ones(no_in))
X = X_dist.sample(torch.Size([n]))

# GAM --------------------------------------------------------------------------
from Tensorflow.Effects.bspline import get_design

no_basis = 20
X_dist = td.Uniform(-10., 10)
X = X_dist.sample(torch.Size([n]))
Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=no_basis), dtype=torch.float32, requires_grad=False)

# (1) unbijected
gam = M.GAM(no_basis=no_basis, order=1, bijected=False)
gam.reset_parameters()
gam.true_model = gam.vec

# (2) bijected
gam_bij = M.GAM(no_basis=no_basis, order=1, bijected=True)
gam_bij.true_model = gam_bij.vec

# (3) penK


# (BNN) ------------------------------------------------------------------------
hunits = [1, 10, 1]
bnn = M.BNN(hunits)
bnn.true_model = bnn.vec

# (ShrinkageBNN) ---------------------------------------------------------------
# (1) seperated
shrinkageBNN = M.ShrinkageBNN(hunits, shrinkage='glasso', seperated=True)
shrinkageBNN.true_model = shrinkageBNN.vec

# (2) not seperated
shrinkageBNN_sep = M.ShrinkageBNN(hunits, shrinkage='glasso', seperated=True)
shrinkageBNN_sep.true_model = shrinkageBNN_sep.vec
