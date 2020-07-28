import torch
import torch.nn as nn
import torch.distributions as td
import hamiltorch.util as util
from functools import partial

from Pytorch.Models.GAM import GAM
from Pytorch.Models.ShrinkageBNN import ShrinkageBNN
from Tensorflow.Effects.bspline import get_design


class Structured_BNN(ShrinkageBNN):

    def __init__(self, gam_param={'xgrid': (0, 10, 0.5), 'order': 1, 'no_basis': 20,
                                  'no_out': 1, 'activation': nn.Identity()},
                 hunits=[2, 10, 1], shrinkage='glasso', activation=nn.ReLU(), final_activation=nn.ReLU()):

        super().__init__(hunits, activation, final_activation, shrinkage)
        self.gam = GAM(**gam_param)

    def sample(self, seperate=True):
        """
        Sample the entire Prior Model
        :param seperate: True indicates whether or not, the true underlying model
        has a joint or disjoint effect (i.e. shrinkage or no shrinkage)
        :return: list of randomly sampled parameters
        """
        if seperate:
            pass
        else:
            pass

    def __call__(self, X, Z, alpha=1.):
        """

        :param X: Designmatrix (batch shaped tensor)
        :param Z: Bspline Design matrix expansion of the first column in X
        (batch shaped tensor)
        :param alpha:
        :return:
        """
        if alpha < 0. or alpha > 1.:
            raise ValueError('alpha exceeded (0,1) interval')
        return self.layers(X) + alpha * self.gam(Z)

    def prior_log_prob(self):
        """surrogate for the hidden layers' prior log prob"""
        bnn_log_prob = sum([h.prior_log_prob().sum() for h in self.layers])
        return bnn_log_prob + sum(self.gam.prior_log_prob())

    def likelihood(self, X, Z, alpha, sigma=torch.tensor(1.)):
        return td.Normal(self(X, Z, alpha), sigma)

    def log_prob(self, X, Z, y, vec):
        """SG log_prob"""
        self.vec_to_attrs_structured()


        return self.prior_log_prob() + self.likelihood(X, Z, self.layers[0].alpha).log_prob(y)

    def closure_log_prob(self, X=None, Z=None, y=None):
        """full_dataset log_prob decorator"""
        print('Setting up "Full Dataset" mode')
        self.log_prob = partial(self.log_prob, X, Z, y)

    def vec_to_attrs_structured(self, vec):
        """parsing the vec to the appropriate places (inplace)"""
        self.gam.vec_to_attrs(vec[:self.gam.n_params])
        self.vec_to_attrs(vec[self.gam.n_params:])


if __name__ == '__main__':
    no_basis = 20
    X_dist = td.Uniform(-10., 10)
    X = X_dist.sample(torch.Size([100]))
    X.detach()
    Z = torch.tensor(get_design(X.numpy(), degree=2, no_basis=no_basis), dtype=torch.float32, requires_grad=False)
    Z.detach()

    Structured_BNN(shrinkage='glasso')
    Structured_BNN(shrinkage='gspike')
    ghorse = Structured_BNN(shrinkage='ghorse')

    # check if the model parameters are parsed correctly
    ghorse.vec_to_attrs(torch.cat([i * torch.ones(h.n_params) for i, h in enumerate(ghorse.layers)]))
    [h.W for h in ghorse.layers]  # notice this does not cover the optional bias
    ghorse.vec_to_attrs(torch.cat(torch.ones(80)))
    ghorse(X)

    # check sampling

    # check forward path

    # check log_prob(X, Z, y)

    # check sampling ability.

    util.flatten(ghorse)
