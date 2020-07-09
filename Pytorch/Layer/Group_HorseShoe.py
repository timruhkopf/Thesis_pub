import torch
import torch.nn as nn
import torch.distributions as td

from Pytorch.Layer.Hidden import Hidden
from Pytorch.Layer.Group_lasso import Group_lasso


class Group_HorseShoe(Hidden):
    prior_log_prob = Group_lasso.prior_log_prob
    vec_to_attrs = Group_lasso.vec_to_attrs

    def __init__(self, no_in, no_out, bias=True, activation=nn.ReLU()):
        nn.Module.__init__(self)
        self.bias = False  # to meet the Hidden default standard!
        self.no_in = no_in
        self.no_out = no_out
        self.bias = bias
        self.activation = activation

        self.dist = dict()

        # Group lasso structure of W
        self.W_ = nn.Parameter(torch.Tensor(self.no_in, self.no_out))
        self.W = None
        # self.dist['W_shrinked'] = td.Normal(torch.zeros(self.no_in), self.tau)
        # FIXME: check sigma dependence in W_shrinked: \beta_g | tau²_g, sigma² ~ MVN
        self.dist['W'] = td.Normal(torch.zeros(self.no_in * (self.no_out - 1)), 1.)

        # add optional bias
        if bias:
            self.b_ = nn.Parameter(torch.Tensor(self.no_out))
            self.b = None
            self.tau_b = 1.
            self.b = None
            self.dist['b'] = td.Normal(0., 1.)

    def update_distributions(self):
        pass

    def reset_parameters(self):
        pass

    @property
    def alpha(self):
        """attribute in interval [0,1], which decides upon the degree of how much
        weight gam gets in likelihoods'mu= bnn() + alpha *gam()"""
        raise NotImplementedError('horse\'s alpha is not yet implemented')


if __name__ == '__main__':
    pass
