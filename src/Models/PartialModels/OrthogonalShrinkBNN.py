import torch
import torch.nn as nn
import torch.distributions as td

from src.Util.Util_Model import Util_Model
from src.Models.ShrinkageBNN import ShrinkageBNN
from src.Models.OrthogonalBNN import OrthogonalBNN
from src.Util.Util_bspline import get_design


class OrthogonalShrinkBNN(OrthogonalBNN, Util_Model):
    def __init__(self, hunits=[2, 3, 1], shrinkage='glasso',
                 activation=nn.ReLU(), final_activation=nn.ReLU(),
                 seperated=True, bijected=True,
                 no_basis=20):
        nn.Module.__init__(self)
        self.no_basis = no_basis
        self.hunits = hunits
        self.activation = activation
        self.final_activation = final_activation
        self.seperated = seperated
        self.bijected = bijected

        # define the model components
        self.bnn = ShrinkageBNN(hunits, activation, final_activation, shrinkage,
                                seperated=seperated, bijected=bijected, prior='flat')
        self.gam = self.gam_layer['fix_nullspace'](no_basis=no_basis, bijected=bijected)

        self.reset_parameters()


if __name__ == '__main__':
    from copy import deepcopy

    # parametrize the basic datamodel
    no_basis = 20
    no_in = 2
    no_out = 1
    n = 1000

    # generate data
    X_dist = td.Uniform(torch.ones(no_in) * (-10.), torch.ones(no_in) * 10.)
    X = X_dist.sample(torch.Size([n])).view(n, no_in)
    Z = torch.tensor(get_design(X[:, 0].numpy(), degree=2, no_basis=no_basis),
                     dtype=torch.float32)

    h = OrthogonalShrinkBNN(hunits=[2, 10, 5, 1], shrinkage='ghorse', bijected=True)
    h.reset_parameters(True)
    h.true_model = deepcopy(h.state_dict())

    y = h.likelihood(X, Z).sample()

    import matplotlib

    matplotlib.use('TkAgg')
    h.plot(X[:500], Z[:500], y[:500])

    # check sampling ability.
    from src.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 1000, 10000

    trainset = TensorDataset(X, Z, y)
    trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']

    # h.reset_parameters(False)
    # glasso.plot(X_joint, y)

    # torch.autograd.set_detect_anomaly(True)
    h.reset_parameters()
    sampler = Sampler(h, epsilon=0.001, L=2)
    sampler.sample(trainloader, burn_in, n_samples)

    print(sampler.chain_mat)
    # print(sampler.model.a)
    import random

    h.plot(X[:100], Z[:100], y[:100], chain=random.sample(sampler.chain, 100),
           **{'title': 'structuredBNN'})

    h.plot(X[:100], Z[:100], y[:100], chain=list(sampler.chain[-1]))
