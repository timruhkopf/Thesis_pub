import torch
import torch.distributions as td

import Pytorch.Models as M

no_in = 2
no_units = 1
n = 100
X_dist = td.Uniform(-10. * torch.ones(no_in), 10 * torch.ones(no_in))
X = X_dist.sample(torch.Size([n]))

# GAM --------------------------------------------------------------------------



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
