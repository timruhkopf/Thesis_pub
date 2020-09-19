import torch
import torch.nn as nn
import torch.distributions as td
from copy import deepcopy

from Pytorch.Util.DistributionUtil import LogTransform
from Pytorch.Models.GAM import GAM
from Tensorflow.Effects.bspline import get_design, diff_mat1D


class GAM_Wood(GAM):

    def define_model(self):
        # set up tau & W
        GAM.define_model(self)

        # nullspace penalty "variance"
        self.l0 = nn.Parameter(torch.Tensor(1))
        self.dist['l0'] = td.Gamma(2., 2.)

        if self.bijected:
            self.dist['l0'] = td.TransformedDistribution(self.dist['l0'], LogTransform())

        self.l0.data = self.dist['l0'].sample()

        # construct double (null space) penalty
        self.K = torch.tensor(diff_mat1D(self.no_basis, self.order)[1], dtype=torch.float32, requires_grad=False)
        eig_val, eig_vec = torch.eig(self.K, eigenvectors=True)
        threshold = 1e-2
        self.rangespace = eig_vec[:, eig_val[:, 0] > threshold]
        self.nullspace = eig_vec[:, eig_val[:, 0] < threshold]

        # double penalty - refactored
        # W^T (tau * K) W + W^T (l0 * U0 U0^T) W
        # = W^T (tau * K + l0 * U0 U0^T) W
        # = W^T (tau * K + l0 * U0 U0^T) W
        # = W^T (tau * K + l0 * null*J) W  since K is rank deficient by 1 and self.nullspace is null * torch.ones()
        self.dist['W'] = td.MultivariateNormal(torch.zeros(self.no_basis), self.null_cov.detach())

    @property
    def null_cov(self):
        if self.bijected:
            tau = self.dist['tau'].transforms[0]._inverse(self.tau)
            l0 = self.dist['l0'].transforms[0]._inverse(self.l0)
        else:
            tau = self.tau.data
            l0 = self.l0.data
        return torch.inverse(tau * self.K + l0 * self.nullspace[0] ** 2)

    def reset_parameters(self, tau=torch.tensor([1.])):
        self.l0.data = self.dist['l0'].sample()
        GAM.reset_parameters(self, tau)

    def update_distributions(self):
        self.dist['W'].covariance_matrix = self.null_cov

    def prior_log_prob(self):
        return self.dist['W'].log_prob(self.W) + self.dist['tau'].log_prob(self.tau) + self.dist['l0'].log_prob(self.l0)

    # @property
    # def double_penalty(self):
    #     # for computational convenience- since K is deficient by only 1,
    #     # null @ null.t() = null[0]**2 * J
    #
    #     self.W.t() @ (self.tau * self.K + self.l0 * self.nullspace[0] ** 2 * self.J) @ self.W
    #     return self.tau * self.W.t() @ self.K @ self.W + \
    #            self.l0 * self.W.t() @ self.nullspace @ self.nullspace.t() @ self.W
    #


if __name__ == '__main__':

    # dense example
    no_basis = 20
    n = 1000
    X_dist = td.Uniform(-10., 10)
    X = X_dist.sample(torch.Size([n]))
    Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=no_basis),
                     dtype=torch.float32, requires_grad=False)

    gam = GAM_Wood(no_basis=no_basis, order=1, activation=nn.Identity(), bijected=True)
    # gam = GAM(no_basis=no_basis, order=1, activation=nn.Identity(), bijected=False)

    gam.reset_parameters(tau=torch.tensor([0.0001]))
    # gam.reset_parameters(tau=torch.tensor([0.01]))
    gam.true_model = deepcopy(gam.state_dict())
    y = gam.likelihood(Z).sample()

    gam.reset_parameters(tau=torch.tensor([1.]))

    gam.plot(X, y)

    gam.forward(Z)
    gam.prior_log_prob()

    from Pytorch.Samplers.LudwigWinkler import SGNHT, SGLD, MALA
    from Pytorch.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    import matplotlib
    import random
    import os

    matplotlib.use('Agg')  # 'TkAgg' for explicit plotting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.get_device_name(0)
    Z.to(device)
    y.to(device)

    from pathlib import Path

    home = str(Path.home())
    if '/PycharmProjects' in __file__:
        # file is on local machine
        home += '/PycharmProjects'
    path = home + '/Thesis/Pytorch/Experiments/Results_GAM_Wood/'
    if not os.path.isdir(path):
        os.mkdir(path)

    sampler_name = ['SGNHT', 'SGLD', 'MALA', 'RHMC', 'SGRLD', 'SGRHMC'][3]
    model = gam

    # Setting up the parameters  -----------------------------------------------
    sg_batch = 100
    for rep in range(3):
        for L in [1, 2, 3]:
            for eps in np.arange(0.007, 0.04, 0.003):
                model.reset_parameters()  # initialization
                name = '{}_{}_{}_{}'.format(sampler_name, str(eps), str(L), str(rep))

                # MAYBE DO STOCHASTIC GRADIENT DESCENT
                trainset = TensorDataset(Z, y)
                sampler_param = dict(
                    epsilon=eps, num_steps=1000, burn_in=1000,
                    pretrain=False, tune=False, num_chains=1)

                if sampler_name in ['SGNHT', 'RHMC', 'SGRHMC']:
                    sampler_param.update(dict(L=L))

                if sampler_name == 'SGRHMC':
                    sampler_param.update(dict(alpha=0.2))

                if 'SG' in sampler_name:
                    batch_size = sg_batch
                else:
                    batch_size = X.shape[0]

                # TODO find a reasonable init
                # # SGD optimizing for reasonable init
                # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                # dataset = TensorDataset(Z, y)
                #
                # for input, target in dataset:
                #     optimizer.zero_grad()
                #     output = model(input)
                #     loss = gam.my_log_prob(output, target)
                #     loss.backward()
                #     optimizer.step()

                # Setting up the sampler & sampling
                if sampler_name in ['SGNHT', 'SGLD', 'MALA']:  # geoopt based models
                    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
                    Sampler = {'SGNHT': SGNHT,  # step_size, hmc_traj_length
                               'MALA': MALA,  # step_size
                               'SGLD': SGLD  # step_size
                               }[sampler_name]
                    sampler = Sampler(model, trainloader, **sampler_param)
                    try:
                        sampler.sample()
                        print(sampler.chain[:3])
                        print(sampler.chain[-3:])

                        # Visualize the resulting estimation -------------------------
                        sampler.model.plot(X[:100], y[:100], sampler.chain[:30], path=path + name)
                        sampler.model.plot(X[:100], y[:100], random.sample(sampler.chain, 30), path=path + name)
                        matplotlib.pyplot.close('all')
                    except Exception as error:
                        print(name, 'failed')
                        sampler.model.plot(X[:100], y[:100], path=path + 'failed_' + name)
                        print(error)

                elif sampler_name in ['RHMC', 'SGRLD', 'SGRHMC']:
                    n_samples = sampler_param.pop('num_steps')
                    burn_in = sampler_param.pop('burn_in')
                    sampler_param.pop('pretrain')
                    sampler_param.pop('tune')
                    sampler_param.pop('num_chains')

                    trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

                    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
                               'SGRLD': myRSGLD,  # epsilon
                               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
                               }['RHMC']
                    sampler = Sampler(model, **sampler_param)
                    try:
                        sampler.sample(trainloader, burn_in, n_samples)

                        print(sampler.chain[:3])
                        print(sampler.chain[-3:])

                        # Visualize the resulting estimation -------------------------

                        sampler.model.plot(X[:100], y[:100], sampler.chain[:30], path=path + name)
                        sampler.model.plot(X[:100], y[:100], random.sample(sampler.chain, 30), path=path + name)
                        matplotlib.pyplot.close('all')
                    except Exception as error:
                        print(name, 'failed')
                        sampler.model.plot(X[:100], y[:100], path=path + 'failed_' + name)
                        print(error)
