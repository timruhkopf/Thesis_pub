import torch
import torch.nn as nn
import torch.distributions as td

import Pytorch.Layer as L


no_in = 2
no_units = 1
n = 100
X_dist = td.Uniform(-10.* torch.ones(no_in), 10* torch.ones(no_in))
X = X_dist.sample(torch.Size([n]))

# Hidden layer regression ------------------------------------------------------
reg_Hidden = L.Hidden(no_in, no_units, bias=False, activation=nn.Identity())
reg_Hidden.true_model = reg_Hidden.vec

reg_Hidden_Prob = L.Hidden_Probmodel()
reg_Hidden_Prob.true_model = reg_Hidden_Prob.vec
y_Hidden = reg_Hidden.likelihood(X)

# Group Lasso regression -------------------------------------------------------
# (1) unbijected
reg_lasso = L.Group_lasso()
reg_lasso_Prob = L.Group_lasso_Probmodel()
y_lasso = reg_lasso.likelihood(X)

# (2) bijected


# Group Horse Shoe regression --------------------------------------------------