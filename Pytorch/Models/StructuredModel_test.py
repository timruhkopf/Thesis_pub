import torch
import torch.distributions as td

# slicing works gracefully
# a = torch.ones([10, 10], requires_grad=True)
# b = torch.ones([10, 3], requires_grad=True)
# c = torch.ones([10, 7], requires_grad=True)
#
# d = (a[:, :3] ** 2 - b ** 2).sum()
# e = (a[:, 3:] - c).sum()
# L = d + e
# L.backward()


from Pytorch.Models.GAM import GAM
from Pytorch.Models.ShrinkageBNN import ShrinkageBNN
from Pytorch.Models.BNN import BNN
from Pytorch.Layer.Group_lasso import Group_lasso


import torch.nn as nn


class H(nn.Module):
    def __init__(self):
        super().__init__()
        # both models are failing
        # self.bnn = ShrinkageBNN(hunits=[2, 3, 1]) # group_lasso is the first layer here!
        self.bnn = Group_lasso(2, 1, bijected=True)

        # these models run with no problem at all
        # self.bnn = BNN(hunits=[2, 3, 1])
        self.gam = GAM(no_basis=20)

        self.bnn.reset_parameters()
        self.gam.reset_parameters()

    def forward(self, X):
        return self.bnn.forward(X[:, :2]) + self.alpha * self.gam.forward(X[:, 2:])

    @property
    def alpha(self):
        if isinstance(self.bnn, BNN):
            tau = self.bnn.layers[0].W[0, 0].clone().detach()
        elif isinstance(self.bnn, Group_lasso):
            tau = self.bnn.W[0, 0].clone().detach()  # notice, that this should be actually tau

        tau.requires_grad_(False)
        return tau

    def prior_log_prob(self):
        """surrogate for the hidden layers' prior log prob"""
        return self.bnn.prior_log_prob() + sum(self.gam.prior_log_prob())

    def likelihood(self, X, sigma=torch.tensor(1.)):
        """:param X: joint matrix, first cols are classic design, the following are
        the bspline extension desing marix"""
        return td.Normal(self.forward(X), sigma)

    def log_prob(self, X, y):
        return self.prior_log_prob() + self.likelihood(X).log_prob(y).sum()


if __name__ == '__main__':
    X = torch.ones([10, 22])

    h = H()
    # L = h(X).sum()
    # L.backward()

    # h.alpha
    # h.h1.layers

    y = h.likelihood(X).sample([len(X)])

    # check sampling ability.
    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 100, 1000

    trainset = TensorDataset(X, y)
    trainloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']

    # h.reset_parameters(False)
    # glasso.plot(X_joint, y)

    torch.autograd.set_detect_anomaly(True)
    sampler = Sampler(h, epsilon=0.001, L=2)
    sampler.sample(trainloader, burn_in, n_samples)

    [(name, p.grad) for name, p in self.model.named_parameters()]
