import torch
import torch.nn as nn

from src.Models.OrthogonalBNN import OrthogonalBNN
from src.Layer.Hidden import Hidden
from src.Layer.GAM import GAM
from src.Layer.GAM_fix_var import GAM_fix_var
from copy import deepcopy


class Orth_GAM_Reg(OrthogonalBNN):
    gam_layer = {
        'fix_nullspace': GAM,
        'fix_variance': GAM_fix_var
    }

    def __init__(self, hunits=[2, 1],
                 activation=nn.ReLU(), final_activation=nn.ReLU(),
                 bijected=True,
                 no_basis=20, gam_type='fix_nullspace'):
        """
        See OrthogonalBNN & Structured for base doc. The purpose of this class
        is to test the orthogonalisation & trainings behaviour. does the Pz_ortho
        work in the training procedure?
        Here, GAM is supposed to estimate the non linear additive part after the
        linear part is estimated by a plain regression.

        :param hunits:
        :param activation:
        :param final_activation:
        :param bijected: whether or not the Gam's variance parameter (only hierarchical parameter in OrthogonalBNN
        should be bijected: see GAM doc source code for details)
        :param no_basis: number J of spline basis functions B_J(x) gathered in Z
        """
        nn.Module.__init__(self)
        self.no_basis = no_basis
        self.hunits = hunits
        self.activation = activation
        self.final_activation = final_activation
        self.bijected = bijected

        # define the model components
        self.bnn = Hidden(bias=False, no_in=hunits[0], no_out=hunits[1], activation=activation)

        if gam_type == 'fix_variance':
            self.gam = self.gam_layer[gam_type](no_basis=no_basis, tau=1.)
        elif gam_type == 'fix_nullspace':
            self.gam = self.gam_layer[gam_type](no_basis=no_basis, bijected=bijected)

        self.reset_parameters()
        self.true_model = deepcopy(self.state_dict())

    def forward(self, X, Z):
        self.Pz_orth = self.orth_projection(X)
        self.Pz_orth.detach()
        return self.bnn(X) + self.gam(self.Pz_orth @ Z)

    def reset_parameters(self, **kwargs):
        """NOTICE, that since Ortho_GAM_REG uses OrthoBNN & StructBNN interface
        with self.bnn = Hidden and Hidden.reset_parameter does not have 'seperated'
        argument, this method has to be overwritten"""
        # Resample the BNN part
        self.bnn.reset_parameters()
        self.gam.reset_parameters(**kwargs)
        self.init_model = deepcopy(self.state_dict())


if __name__ == '__main__':
    import torch.distributions as td

    h = Orth_GAM_Reg(hunits=[2, 1], gam_type='fix_variance', no_basis=10,
                     activation=nn.Identity(), final_activation=nn.Identity())
    n = 1000
    X, Z, y = h.sample_model(n)
    h.reset_parameters()

    # import matplotlib
    # matplotlib.use('TkAgg')
    # h.plot(X, Z, y)

    # chain = []
    # losses = []
    # print('Pz_ortho', h.Pz_orth @ X)
    # for step in range(100):
    #     y_pred = h.forward(X, Z)
    #     MSE = ((y_pred - y) ** 2).sum()
    #     loss = h.log_prob(X, Z, y)
    #     loss.backward()
    #     losses.append(loss)
    #     print('tau\'s gradient: ', h.gam.tau.grad)
    #
    #     with torch.no_grad():
    #
    #         # print('BNN')
    #         # for name, p in h.bnn.named_parameters():
    #         #     print(name, torch.flatten(p.data), 'grad: ', torch.flatten(p.grad),
    #         #           'update: ', torch.flatten(torch.flatten(0.001 * p.grad)))
    #         #
    #         # print('GAM')
    #         for name, p in h.named_parameters():
    #             # print(p.grad)
    #
    #             try:
    #                 if name == 'gam.tau':
    #                     # update tau more sensibly in smaller steps
    #                     print('gam.tau (on R): ', p.data, 'gam.tau_bij (on R+):', h.gam.tau_bij, '\n',
    #                           'grad:', p.grad, 'update: ', 0.0001 * p.grad, '\n')
    #
    #                     print('gam.W.grad:', torch.flatten(h.gam.W.grad), '\n',
    #                           'gam.W.update: ', torch.flatten(0.001 * h.gam.W.grad), '\n\n')
    #                     p -= 0.0001 * p.grad + td.Normal(torch.zeros_like(p), 1.).sample()
    #                 elif name == 'gam.W':
    #                     # default update
    #                     print('gam.W.grad:', torch.flatten(h.gam.W.grad), '\n',
    #                           'gam.W.update: ', torch.flatten(0.001 * h.gam.W.grad), '\n\n')
    #                     p -= 0.001 * p.grad + td.Normal(torch.zeros_like(p), 1.).sample()
    #
    #                 else:
    #                     print(name, ' grad:', torch.flatten(p.grad), '\n',
    #                           name, '.update: ', torch.flatten(0.001 * p.grad), '\n\n')
    #                     p -= 0.001 * p.grad + td.Normal(torch.zeros_like(p), 1.).sample()
    #
    #
    #
    #             except:
    #                 # tau seems to not be include in prior - since it is not in the loss and has no gradient as
    #                 # consequence!
    #                 print(name, ' failed to be updated')
    #                 continue
    #
    #         for p in h.parameters():
    #             chain.append(deepcopy(h.state_dict()))
    #             p.grad = None
    #
    # import matplotlib
    # matplotlib.use('TkAgg')
    # h.plot(X, Z, y, chain=chain)
    # print()
    #

    # check sampling ability.
    from src.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 100, 1000

    trainset = TensorDataset(X, Z, y)
    trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=0)

    Sampler = {'RHMC': myRHMC,  # epsilon, n_steps
               'SGRLD': myRSGLD,  # epsilon
               'SGRHMC': mySGRHMC  # epsilon, n_steps, alpha
               }['RHMC']

    # torch.autograd.set_detect_anomaly(True)
    h.reset_parameters()
    sampler = Sampler(h, epsilon=0.01, L=1)
    sampler.sample(trainloader, burn_in, n_samples)

    print(sampler.chain_mat)
    import random

    h.plot(X[:100], Z[:100], y[:100], chain=random.sample(sampler.chain, 100),
           **{'title': 'orthoBNN'})

    h.plot(X[:100], Z[:100], y[:100], chain=list(sampler.chain[-1:]))
    h.plot(X[:100], Z[:100], y[:100], chain=list(h.init_model))
