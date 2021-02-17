import torch
import torch.nn as nn
import torch.distributions as td

from src.Util.Util_Model import Util_Model
from src.Layer.GAM import GAM
from src.Layer.GAM_fix_var import GAM_fix_var

from src.Models.BNN import BNN
from src.Models.StructuredBNN import StructuredBNN
from src.Util.Util_bspline import get_design

from copy import deepcopy


class OrthogonalBNN(StructuredBNN, Util_Model):
    gam_layer = {
        'fix_nullspace': GAM,
        'fix_variance': GAM_fix_var
    }

    def __init__(self, hunits=[2, 3, 1],
                 activation=nn.ReLU(), final_activation=nn.ReLU(),
                 bijected=True,
                 no_basis=20, gam_type='fix_nullspace'):
        """
        Orhtogonal BNN is a simplified implementation of "A Unified Network Architecture
        for Semi-Structured Deep Distributional Regression".
        The main target of this model is to find an identifiable composition of
        an additive (gam) effect and the remaining interaction effect on top of it
        (including all other variable's main effects)
        Here, the variable x_1 is input to both a GAM and a BNN model.
        The Basis Expansion within GAM of x_1 is {B_j(x_1)}_{j=1}^J = Z
        is used to calculate an orhtogonal projection of Z. This projection in turn
        is used to clean the BNN's penultimate layer's output of the additive function learned by GAM,
        before its output is combined to form y_hat_BNN. The prediction of this model is
        y_hat = y_hat_BNN(X) + y_hat_GAM({B_j(x_1)}_{j=1}^J).

        :param hunits:
        :param activation:
        :param final_activation:
        :param bijected: whether or not the Gam's variance parameter (only hirachical parameter in OrthogonalBNN
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
        self.bnn = BNN(hunits=self.hunits, activation=self.activation,
                       final_activation=self.final_activation, prior='normal')

        if gam_type == 'fix_variance':
            self.gam = self.gam_layer[gam_type](no_basis=no_basis, tau=1.)
        elif gam_type == 'fix_nullspace':
            self.gam = self.gam_layer[gam_type](no_basis=no_basis, bijected=bijected)

        self.reset_parameters()

    @staticmethod
    def orth_projection(X):
        """Orthogonal projection matrix on orth. columnspace of X
        :return  Porthogonal_X: (I - X(X'X)**-1 X')
        """
        return (torch.eye(X.shape[0]) - X @ torch.inverse(X.t() @ X) @ X.t())

    def forward(self, X, Z):
        """the prediction path of the NN (standard nn.Module interface)
        This particular path is taken from "A Unified Network Architecture
        for Semi-Structured Deep Distributional Regression"

        :param X: Input to the BNN
        :param Z: GAM imput
        :return: y_hat vector: prediction of the model
        """

        # calculate the current's Z_trainingdata subset's ortho. Projection matrix
        self.Pz_orth = self.orth_projection(Z)
        self.Pz_orth.detach()

        # penultimate layer
        for l in self.bnn.layers[:-1]:
            X = l(X)

        # remove linear part from gam estimation:
        constrained_penultimate = self.Pz_orth @ X

        return self.gam(Z) + self.bnn.layers[-1].forward(constrained_penultimate)

    def sample_model(self, n):
        X_dist = td.Uniform(torch.ones(self.bnn.no_in) * (-10.), torch.ones(self.bnn.no_in) * 10.)
        X = X_dist.sample(torch.Size([n])).view(n, self.bnn.no_in)
        Z = torch.tensor(get_design(X[:, 0].numpy(), degree=2, no_basis=self.no_basis),
                         dtype=torch.float32)
        self.reset_parameters()
        self.true_model = deepcopy(self.state_dict())

        y = self.likelihood(X, Z).sample()

        return X, Z, y


if __name__ == '__main__':

    # set up model & generate data from it
    bijected = True
    h = OrthogonalBNN(hunits=[2, 3, 1], bijected=bijected, no_basis=5, gam_type='fix_variance')
    n = 1000
    X, Z, y = h.sample_model(n)
    h.reset_parameters()  # reset to random initialisation.

    if bijected and type(h.gam) == type(GAM):
        print(h.gam.tau.dist.transforms[0]._inverse(h.gam.tau))  # original tau
    else:
        print(h.gam.tau)

    # import matplotlib
    #
    # matplotlib.use('TkAgg')
    # # h.plot(X, Z, y)
    # h.plot(X[:500], Z[:500], y[:500])

    # naive training
    # FIXME:
    #     tau's gradient:  tensor([-98.8948])
    #     gam.tau (on R):  tensor([0.3271]) gam.tau_bij (on R+): tensor([1.3870], grad_fn=<ExpBackward>)
    #      grad: tensor([-98.8948]) update:  tensor([-0.0099])
    #     tau's gradient:  tensor([-98.8796])
    #     gam.tau (on R):  tensor([0.3370]) gam.tau_bij (on R+): tensor([1.4007], grad_fn=<ExpBackward>)
    #      grad: tensor([-98.8796]) update:  tensor([-0.0099])
    #     tau's gradient:  tensor([-92.1714])
    #     gam.tau (on R):  tensor([0.3469]) gam.tau_bij (on R+): tensor([1.4147], grad_fn=<ExpBackward>)
    #      grad: tensor([-92.1714]) update:  tensor([-0.0092])
    #     tau's gradient:  tensor([1.7492e+08])
    #     gam.tau (on R):  tensor([0.3561]) gam.tau_bij (on R+): tensor([1.4278], grad_fn=<ExpBackward>)
    #      grad: tensor([1.7492e+08]) update:  tensor([17491.8691])
    #     self.tau_bij
    #     tensor([0.], grad_fn=<ExpBackward>)  <---------- becomes zero since tau's gradient explodes
    #     self.tau
    #     Parameter containing:
    #     tensor([-17491.5137], requires_grad=True)
    # How does the gradient of tau come about: more or less like
    # dL/dtau = dW / dtau(self.W.dist's cov) + dtau_dist / dtau

    chain = []
    losses = []
    for step in range(100):
        y_pred = h.forward(X, Z)
        MSE = ((y_pred - y) ** 2).sum()
        loss = h.log_prob(X, Z, y)
        loss.backward()
        losses.append(loss)
        print('tau\'s gradient: ', h.gam.tau.grad)

        with torch.no_grad():
            print('Pz_ortho', h.Pz_orth @ X)
            # print('BNN')
            # for name, p in h.bnn.named_parameters():
            #     print(name, torch.flatten(p.data), 'grad: ', torch.flatten(p.grad),
            #           'update: ', torch.flatten(torch.flatten(0.001 * p.grad)))
            #
            # print('GAM')
            for name, p in h.named_parameters():
                # print(p.grad)

                try:
                    if name == 'gam.tau':
                        # update tau more sensibly in smaller steps
                        print('gam.tau (on R): ', p.data, 'gam.tau_bij (on R+):', h.gam.tau_bij, '\n',
                              'grad:', p.grad, 'update: ', 0.0001 * p.grad, '\n')

                        print('gam.W.grad:', torch.flatten(h.gam.W.grad), '\n',
                              'gam.W.update: ', torch.flatten(0.001 * h.gam.W.grad), '\n\n')
                        p -= 0.0001 * p.grad + td.Normal(torch.zeros_like(p), 1.).sample()
                    elif name == 'gam.W':
                        # default update
                        print('gam.W.grad:', torch.flatten(h.gam.W.grad), '\n',
                              'gam.W.update: ', torch.flatten(0.001 * h.gam.W.grad), '\n\n')
                        p -= 0.001 * p.grad + td.Normal(torch.zeros_like(p), 1.).sample()

                    else:
                        print(name, ' grad:', torch.flatten(p.grad), '\n',
                              name, '.update: ', torch.flatten(0.001 * p.grad), '\n\n')
                        p -= 0.001 * p.grad + td.Normal(torch.zeros_like(p), 1.).sample()



                except:
                    # tau seems to not be include in prior - since it is not in the loss and has no gradient as
                    # consequence!
                    print(name, ' failed to be updated')
                    continue

            for p in h.parameters():
                chain.append(deepcopy(h.state_dict()))
                p.grad = None

    h.plot(X, Z, y, chain=chain)
    print()

    # check sampling ability.
    from src.Samplers.mygeoopt import myRHMC, mySGRHMC, myRSGLD
    from torch.utils.data import TensorDataset, DataLoader

    burn_in, n_samples = 1000, 1000

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
